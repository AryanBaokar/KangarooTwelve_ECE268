#include "turboshake128.h"
#include <string.h>

/**
 * TurboSHAKE128 sponge construction for KangarooTwelve.
 *
 * TurboSHAKE128 is a XOF (extendable-output function) built on top of
 * Keccak-p[1600, 12]. It operates in two phases:
 *
 *   Absorb: Input bytes are XORed into the first RATE=168 bytes of the
 *           200-byte state. When those bytes fill, keccak_p permutes the
 *           state and absorption continues into the refreshed rate window.
 *
 *   Squeeze: Output bytes are read from the same RATE=168 byte window.
 *            When exhausted, keccak_p runs again to produce more output,
 *            allowing arbitrary-length output (XOF behaviour).
 *
 * The two phases are separated by pad_and_switch, which appends a
 * domain separation byte (encoding the caller's role in the K12 tree)
 * and applies 10*1 padding to complete the final block.
 *
 * Domain bytes used by KangarooTwelve:
 *   0x0B  - leaf chunk (every chunk except the first in multi-block mode)
 *   0x06  - root / final hash (first chunk + all chaining values)
 *   0x07  - single-block message (entire input fits in one chunk)
 */

/* Rate in bytes: r = 1600 - 2*128 = 1344 bits = 168 bytes */
#define RATE 168


/* --------------------------------------------------------------------------
 * Lifecycle
 * -------------------------------------------------------------------------- */

/**
 * Initialise a sponge to the all-zero state.
 * Must be called before any absorb or squeeze operations.
 */
void turboshake128_init(TurboSponge* s) {
    memset(s->state, 0, sizeof(s->state));
    s->position = 0;
}


/* --------------------------------------------------------------------------
 * Absorb phase
 * -------------------------------------------------------------------------- */

/**
 * Absorb `len` bytes of `data` into the sponge.
 *
 * Bytes are XORed one-by-one into the rate portion of the state starting
 * at s->position. When the rate window is full (position == RATE), keccak_p
 * is called to mix the state and the position resets to 0.
 *
 * May be called multiple times; each call continues from where the last
 * left off. Do not call after turboshake128_pad_and_switch.
 */
void turboshake128_absorb(TurboSponge* s, const uint8_t* data, size_t len) {
    uint8_t* state_bytes = (uint8_t*)s->state;
    size_t offset = 0;

    while (offset < len) {
        /* How many bytes fit before the rate window is full */
        size_t take = len - offset;
        if (take > RATE - s->position) {
            take = RATE - s->position;
        }

        for (size_t i = 0; i < take; ++i) {
            state_bytes[s->position++] ^= data[offset + i];
        }
        offset += take;

        /* Full block absorbed — permute and reset position */
        if (s->position == RATE) {
            keccak_p(s->state);
            s->position = 0;
        }
    }
}


/* --------------------------------------------------------------------------
 * Transition: absorb → squeeze
 * -------------------------------------------------------------------------- */

/**
 * Finalise absorption and prepare the sponge for squeezing.
 *
 * Applies TurboSHAKE padding to the current state:
 *   1. XOR the domain separation byte at the current position.
 *      This byte encodes the caller's role in the K12 tree (leaf/root/single).
 *   2. XOR 0x80 into the very last byte of the rate window.
 *      Together with step 1 this is the 10*1 padding rule: the domain byte
 *      acts as the leading '1' bit and 0x80 places a '1' at position RATE-1.
 *   3. Run keccak_p one final time to mix everything before squeezing.
 *
 * After this call, only turboshake128_squeeze may be called on `s`.
 *
 * @param sakura_suffix  Domain byte: 0x0B (leaf), 0x06 (root), 0x07 (single)
 */
void turboshake128_pad_and_switch(TurboSponge* s, uint8_t sakura_suffix) {
    uint8_t* state_bytes = (uint8_t*)s->state;

    /* XOR domain byte at current position, this is the leading pad bit */
    state_bytes[s->position] ^= sakura_suffix;

    /* XOR 0x80 at end of rate window, this is the trailing pad bit */
    state_bytes[RATE - 1] ^= 0x80;

    /* Final permutation; after this the state is ready for squeezing */
    keccak_p(s->state);
    s->position = 0;
}


/* --------------------------------------------------------------------------
 * Squeeze phase
 * -------------------------------------------------------------------------- */

/**
 * Squeeze `len` bytes of output from the sponge into `out`.
 *
 * Reads bytes sequentially from the rate window of the state. When the
 * window is exhausted (position == RATE), keccak_p is called to produce
 * the next block of output. This allows extracting any number of bytes.
 *
 * Must only be called after turboshake128_pad_and_switch.
 */
void turboshake128_squeeze(TurboSponge* s, uint8_t* out, size_t len) {
    uint8_t* state_bytes = (uint8_t*)s->state;
    size_t offset = 0;

    while (offset < len) {
        /* Rate window exhausted — permute to generate next output block */
        if (s->position == RATE) {
            keccak_p(s->state);
            s->position = 0;
        }

        /* How many bytes available before the window is exhausted */
        size_t take = len - offset;
        if (take > RATE - s->position) {
            take = RATE - s->position;
        }

        for (size_t i = 0; i < take; ++i) {
            out[offset + i] = state_bytes[s->position++];
        }
        offset += take;
    }
}