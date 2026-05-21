#pragma once
#include <stdint.h>
#include <stddef.h>
#include "keccak_p.h"

/**
 * TurboSHAKE128 sponge state.
 *
 * state    - 1600-bit Keccak state, laid out as 25 x uint64_t (little-endian lanes).
 * position - Byte offset into the rate window [0, 168).
 *            Tracks how far through the current rate block absorption/squeezing has reached.
 */
typedef struct {
    uint64_t state[25];
    size_t   position;
} TurboSponge;

/**
 * Domain separation bytes for KangarooTwelve.
 * Passed to turboshake128_pad_and_switch to encode the role of each hash call
 * in the K12 tree structure.
 */
#define K12_DOMAIN_LEAF    0x0B  /* Leaf chunk hash (chunks 1..n-1)          */
#define K12_DOMAIN_ROOT    0x06  /* Root hash (first chunk + chaining values) */
#define K12_DOMAIN_SINGLE  0x07  /* Single-block message (input <= 8192 bytes)*/

/**
 * Initialise a TurboSponge to the all-zero state.
 * Must be called before any absorb or squeeze operations.
 */
void turboshake128_init(TurboSponge* s);

/**
 * Absorb `len` bytes of `data` into the sponge.
 * May be called multiple times; state is preserved between calls.
 * Do not call after turboshake128_pad_and_switch.
 */
void turboshake128_absorb(TurboSponge* s, const uint8_t* data, size_t len);

/**
 * Finalise absorption and transition to the squeeze phase.
 * Applies 10*1 padding with the given domain separation byte.
 * Must be called exactly once, after all absorb calls and before squeezing.
 *
 * @param sakura_suffix  One of K12_DOMAIN_LEAF, K12_DOMAIN_ROOT, K12_DOMAIN_SINGLE.
 */
void turboshake128_pad_and_switch(TurboSponge* s, uint8_t sakura_suffix);

/**
 * Squeeze `len` bytes of output from the sponge into `out`.
 * May be called multiple times for streaming output.
 * Must only be called after turboshake128_pad_and_switch.
 */
void turboshake128_squeeze(TurboSponge* s, uint8_t* out, size_t len);