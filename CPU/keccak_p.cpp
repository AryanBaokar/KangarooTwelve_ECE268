#include "keccak_p.h"

#define K12_ROUNDS 12

/**
 * Function for left-rotating a 64 bit word by n bits
 */
static inline uint64_t rot64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

/****************************************************************
 * Pre-computed tables
 ***************************************************************/

/**
 * Rho offsets - Left rotation n for lane (x,y)
 * Lane(0,0) has offset 0 and is never rotated
 * Each bit in lane is shifted by (t+1)(t+2)/2 mod 64
 * Indexed as [y][x]
 */
 static const int RHO_OFFSETS[5][5] = {
    { 0, 1, 62, 28, 27},
    {36, 44,  6, 55, 20},
    { 3, 10, 43, 25, 39},
    {41, 45, 15, 21,  8},
    {18,  2, 61, 56, 14}
};

/** 
 * Keccak-p[1600, 12] Round Constants (Indices 12 to 23)
 */
static constexpr uint64_t ROUND_CONSTANTS[K12_ROUNDS] = {
    0x000000008000808BULL, // Round 12
    0x800000000000008BULL, // Round 13
    0x8000000000008089ULL, // Round 14
    0x8000000000008003ULL, // Round 15
    0x8000000000008002ULL, // Round 16
    0x8000000000000080ULL, // Round 17
    0x000000000000800AULL, // Round 18
    0x800000008000000AULL, // Round 19
    0x8000000080008081ULL, // Round 20
    0x8000000000008080ULL, // Round 21
    0x0000000080000001ULL, // Round 22
    0x8000000080008008ULL  // Round 23
};

/**
 * Keccak p is working on 1600 bit state with 12 rounds.
 * The 1600 bit state is organized as [5x5x64].
 * The state A[x,y,z] has x rows, y columns, and z lanes.
 * Each round contains 5 step mappings -
 * 1. Theta - XORs each bit with parities of 2 specific columns in state array
 * 2. Rho   - Rotates the bits of each of the 25 lanes by a fixed offset.
 * 3. Pi    - Rearranges position of the lanes
 * 4. Chi   - Applies non-linear bitwise op to each row
 * 5. Iota  - Modifies the bits of a single lane using round constant to break symmetry.
 * */ 

void keccak_p(uint64_t state[25])
{
    for(int rnd = 0; rnd < K12_ROUNDS; ++rnd)
    {
        // Step 1 : Theta
        uint64_t C[5];
        uint64_t D[5];

        for (int x = 0; x < 5; ++x) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }

        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ rot64(C[(x + 1) % 5], 1);
        }

        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 5; ++x) {
                state[x + 5 * y] ^= D[x];
            }
        }

        // Step 2 & 3 : Rho & Pi
        uint64_t next_state[25];
        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 5; ++x) {
                int x_old = (x + 3 * y) % 5;
                int y_old = x;
                int shift = RHO_OFFSETS[y_old][x_old];
                
                next_state[x + 5 * y] = rot64(state[x_old + 5 * y_old], shift);
            }
        }

        // Step 4 : Chi
        for (int y = 0; y < 5; ++y) {
            uint64_t T[5];
            for (int x = 0; x < 5; ++x) {
                T[x] = next_state[x + 5 * y];
            }
            for (int x = 0; x < 5; ++x) {
                state[x + 5 * y] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
            }
        }

        // Step 5 : Iota
        state[0] ^= ROUND_CONSTANTS[rnd];
    }
}