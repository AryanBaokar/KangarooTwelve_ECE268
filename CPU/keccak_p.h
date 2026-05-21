#pragma once
#include <cstdint>

/**
 * Keccak-p[1600, 12] permutation.
 *
 * This is the core cryptographic primitive underlying KangarooTwelve.
 * It is a reduced-round variant of the Keccak-f[1600] permutation used
 * in SHA-3, running only the last 12 of the standard 24 rounds.
 *
 * The 1600-bit state is passed as a flat array of 25 uint64_t lanes,
 * laid out in row-major order: state[x + 5*y] corresponds to lane A[x,y]
 * in the spec. Lanes are stored in little-endian byte order.
 *
 * The permutation is applied in-place: the caller's state array is
 * overwritten with the permuted result. There is no return value.
 *
 * Called by the TurboSHAKE128 sponge construction at the end of each
 * absorbed block and at the start of each squeezed block.
 *
 * @param state  25-element array of uint64_t representing the 1600-bit
 *               Keccak state. Modified in-place.
 */
void keccak_p(uint64_t state[25]);