#include "k12.h"
#include "turboshake128.h"
#include <string.h>
#include <stdlib.h>

/**
 * KangarooTwelve implementation  -  RFC 9861
 *
 * Tree structure ("leaves stapled to a pole"):
 *
 *   Input S = M || C || length_encode(|C|)
 *   S is split into n chunks of 8192 bytes (last chunk may be shorter).
 *
 *   Single-block path  (|S| <= 8192):
 *     output = F( S || 0x07, L )
 *
 *   Multi-block path   (|S| >  8192):
 *     For each leaf chunk S_i (i = 1..n-1):
 *       CV_i = F( S_i || 0x0B, 32 )       <-- leaf hash, 32-byte chaining value
 *
 *     FinalNode = S_0
 *              || 0x03 0x00 0x00 0x00 0x00 0x00 0x00 0x00   (Sakura hop marker)
 *              || CV_1 || ... || CV_(n-1)
 *              || length_encode(n-1)
 *              || 0xFF 0xFF
 *
 *     output = F( FinalNode || 0x06, L )  <-- root hash
 *
 * F is TurboSHAKE128: absorb the data, absorb the domain byte via
 * pad_and_switch, then squeeze L bytes. The domain byte is the last
 * byte of the input in the spec's pseudocode; here it is passed
 * separately to turboshake128_pad_and_switch.
 */

/* Chunk size: 8192 bytes = 2^13 bytes */
#define CHUNK_SIZE 8192UL

/* Chaining value length: 32 bytes = 256 bits (= capacity / 8) */
#define CV_LEN 32

/*
 * Sakura hop marker appended to S_0 before the chaining values.
 * Represents the bit-string '11' followed by 62 zero bits, encoded
 * in little-endian byte order as the 64-bit value 0x03.
 */
static const uint8_t HOP_MARKER[8] = { 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };


/* -------------------------------------------------------------------------
 * length_encode(x)
 *
 * Encodes a non-negative integer x as a byte string:
 *   x_(n-1) || ... || x_0 || n
 * where x = sum(256^i * x_i) and n is the minimum number of bytes needed.
 *
 * Special case: length_encode(0) = 0x00  (n=0, append length byte 0)
 *
 * Examples from spec:
 *   length_encode(0)     = 00
 *   length_encode(12)    = 0C 01
 *   length_encode(65538) = 01 00 02 03
 *
 * The encoded bytes are written into `buf` (caller supplies at least 9 bytes).
 * Returns the number of bytes written.
 * ------------------------------------------------------------------------- */
static size_t length_encode(uint64_t x, uint8_t buf[9])
{
    if (x == 0) {
        buf[0] = 0x00; /* length byte, n=0 */
        return 1;
    }

    /* Collect big-endian bytes of x into a temporary array */
    uint8_t tmp[8];
    int n = 0;
    uint64_t v = x;
    while (v > 0) {
        tmp[n++] = (uint8_t)(v & 0xFF);
        v >>= 8;
    }
    /* tmp[0..n-1] are the bytes in little-endian order; reverse to big-endian */
    for (int i = 0; i < n; ++i) {
        buf[i] = tmp[n - 1 - i];
    }
    /* Append the count byte n */
    buf[n] = (uint8_t)n;
    return (size_t)(n + 1);
}


/* -------------------------------------------------------------------------
 * leaf_hash  -  compute CV_i = F( chunk || 0x0B, 32 )
 *
 * Hashes exactly `len` bytes from `chunk` into the 32-byte chaining value
 * stored at `cv_out` using the leaf domain byte 0x0B.
 * ------------------------------------------------------------------------- */
static void leaf_hash(const uint8_t* chunk, size_t len, uint8_t cv_out[CV_LEN])
{
    TurboSponge s;
    turboshake128_init(&s);
    turboshake128_absorb(&s, chunk, len);
    turboshake128_pad_and_switch(&s, K12_DOMAIN_LEAF);
    turboshake128_squeeze(&s, cv_out, CV_LEN);
}


/* -------------------------------------------------------------------------
 * kangaroo_twelve  -  top-level entry point
 *
 * Builds S = M || C || length_encode(|C|), determines single vs multi-block
 * path, hashes accordingly, and writes L bytes to `output`.
 * ------------------------------------------------------------------------- */
void kangaroo_twelve(
    const uint8_t* M, size_t M_len,
    const uint8_t* C, size_t C_len,
    uint8_t*       output, size_t L)
{
    /* --- Encode the customization string suffix ---
     * S = M || C || length_encode(|C|)
     * We don't materialise S in full; instead we feed its parts into the
     * sponge incrementally where possible. The suffix is always short. */
    uint8_t c_enc[9];
    size_t  c_enc_len = length_encode((uint64_t)C_len, c_enc);

    /* Total length of S */
    size_t S_len = M_len + C_len + c_enc_len;

    /* --- Single-block path: |S| <= 8192 --- */
    if (S_len <= CHUNK_SIZE) {
        TurboSponge s;
        turboshake128_init(&s);
        if (M_len > 0) turboshake128_absorb(&s, M,     M_len);
        if (C_len > 0) turboshake128_absorb(&s, C,     C_len);
        turboshake128_absorb(&s, c_enc, c_enc_len);
        turboshake128_pad_and_switch(&s, K12_DOMAIN_SINGLE);
        turboshake128_squeeze(&s, output, L);
        return;
    }

    /* --- Multi-block path: |S| > 8192 ---
     *
     * We read S as a logical stream: M, then C, then c_enc.
     * S_0 is the first 8192 bytes; S_1..S_(n-1) are the leaf chunks.
     *
     * Strategy:
     *   1. Begin absorbing the root sponge with S_0.
     *   2. Append the hop marker to the root sponge.
     *   3. For each leaf chunk S_i (i >= 1):
     *        a. Extract the chunk bytes from the logical stream.
     *        b. Compute CV_i = leaf_hash(S_i).
     *        c. Absorb CV_i into the root sponge.
     *   4. Absorb length_encode(n-1) and 0xFF 0xFF into the root sponge.
     *   5. Finalize root sponge with domain 0x06 and squeeze L bytes.
     *
     * To avoid a large allocation for S, we read the logical stream
     * (M || C || c_enc) in CHUNK_SIZE windows using a helper that
     * copies bytes from the three segments.
     */

    /* Number of chunks n = ceil(S_len / 8192) */
    size_t n = (S_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    /* Helper: logical byte at position pos in S = M || C || c_enc */
    /* We'll use a cursor instead of random access. */

    /* --- Root sponge --- */
    TurboSponge root;
    turboshake128_init(&root);

    /* We read the logical stream with a simple three-segment cursor */
    const uint8_t* seg[3]     = { M,     C,     c_enc     };
    size_t         seg_len[3] = { M_len, C_len, c_enc_len };
    int    seg_idx = 0;     /* which segment we're in */
    size_t seg_off = 0;     /* offset within current segment */

    /* absorb_from_stream: pull `need` bytes from the logical stream into `dst`.
     * Returns the number of bytes actually available (== need unless stream ends). */
    /* Implemented as an inline lambda-style block using a scratch buffer below. */

    /* Scratch buffer for extracting one chunk at a time from the stream */
    uint8_t chunk_buf[CHUNK_SIZE];

    for (size_t chunk_idx = 0; chunk_idx < n; ++chunk_idx) {
        /* How many bytes does this chunk contain? */
        size_t chunk_start = chunk_idx * CHUNK_SIZE;
        size_t chunk_end   = chunk_start + CHUNK_SIZE;
        if (chunk_end > S_len) chunk_end = S_len;
        size_t this_chunk_len = chunk_end - chunk_start;

        /* Extract this_chunk_len bytes from the logical stream into chunk_buf */
        size_t filled = 0;
        while (filled < this_chunk_len && seg_idx < 3) {
            size_t avail = seg_len[seg_idx] - seg_off;
            size_t want  = this_chunk_len - filled;
            size_t take  = (want < avail) ? want : avail;
            memcpy(chunk_buf + filled, seg[seg_idx] + seg_off, take);
            filled  += take;
            seg_off += take;
            if (seg_off == seg_len[seg_idx]) {
                seg_idx++;
                seg_off = 0;
            }
        }

        if (chunk_idx == 0) {
            /* S_0 goes into the root sponge, followed by the hop marker */
            turboshake128_absorb(&root, chunk_buf, this_chunk_len);
            turboshake128_absorb(&root, HOP_MARKER, sizeof(HOP_MARKER));
        } else {
            /* Leaf chunk: hash it into a 32-byte chaining value, feed to root */
            uint8_t cv[CV_LEN];
            leaf_hash(chunk_buf, this_chunk_len, cv);
            turboshake128_absorb(&root, cv, CV_LEN);
        }
    }

    /* Append length_encode(n-1) and the Sakura terminator 0xFF 0xFF */
    uint8_t n_enc[9];
    size_t  n_enc_len = length_encode((uint64_t)(n - 1), n_enc);
    turboshake128_absorb(&root, n_enc, n_enc_len);

    const uint8_t terminator[2] = { 0xFF, 0xFF };
    turboshake128_absorb(&root, terminator, 2);

    /* Finalize and squeeze */
    turboshake128_pad_and_switch(&root, K12_DOMAIN_ROOT);
    turboshake128_squeeze(&root, output, L);
}
