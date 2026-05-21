#pragma once
#include <stdint.h>
#include <stddef.h>

/**
 * KangarooTwelve (K12) — RFC 9861 / draft-irtf-cfrg-kangarootwelve
 *
 * An eXtendable Output Function (XOF) built on TurboSHAKE128 and a
 * "leaves stapled to a pole" Sakura-compatible tree hash mode.
 *
 * The input S = M || C || length_encode(|C|) is split into 8192-byte
 * chunks. Each chunk after the first is hashed independently (leaf hash)
 * into a 32-byte chaining value. The first chunk plus all chaining values
 * are then hashed together to produce the final output (root hash).
 *
 * When |S| <= 8192 bytes the tree degenerates to a single sponge call
 * with no overhead.
 */

/**
 * Hash message M with optional customization string C, producing L output bytes.
 *
 * @param M          Message bytes (may be NULL if M_len == 0)
 * @param M_len      Length of M in bytes
 * @param C          Customization string bytes (may be NULL if C_len == 0)
 * @param C_len      Length of C in bytes
 * @param output     Buffer to receive output bytes
 * @param L          Number of output bytes to produce (must be >= 1)
 */
void kangaroo_twelve(
    const uint8_t* M, size_t M_len,
    const uint8_t* C, size_t C_len,
    uint8_t*       output, size_t L
);
