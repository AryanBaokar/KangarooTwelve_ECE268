#include "keccak_p.h"

/**
 * Keccak-p[1600, 12] permutation — implementation.
 *
 * Keccak-p[1600, 12] is the permutation Keccak-f[1600] restricted to its
 * last 12 rounds (rounds 12 through 23 in the full 24-round schedule).
 * It operates on a 1600-bit state organised as a 5x5 array of 64-bit lanes.
 *
 * Each of the 12 rounds applies five step mappings in sequence:
 *   Theta — linear diffusion across columns
 *   Rho   — intra-lane bit rotation
 *   Pi    — inter-lane transposition
 *   Chi   — non-linear row mixing (the only non-linear step)
 *   Iota  — round constant injection to break symmetry
 *
 * State layout:
 *   The spec defines the state as A[x][y][z] with x,y in {0..4} and z in
 *   {0..63}. In this implementation the 25 lanes are stored as a flat
 *   array indexed by (x + 5*y), so lane A[x,y] = state[x + 5*y].
 *   Within each lane, bit z is stored at position z of the uint64_t
 *   (little-endian bit ordering).
 */

/* Number of rounds for the K12 variant of Keccak-p */
#define K12_ROUNDS 12

/* -------------------------------------------------------------------------
 * rot64 — 64-bit left rotation
 *
 * Rotates the bits of x left by n positions. Used by the Rho step to
 * shift each lane by its pre-computed offset. Implemented as a pair of
 * shifts ORed together; modern compilers recognise this idiom and emit a
 * single rotate instruction (e.g. ROL on x86).
 * ------------------------------------------------------------------------- */
static inline uint64_t rot64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

/****************************************************************
 * Pre-computed tables
 ***************************************************************/

/**
 * RHO_OFFSETS — per-lane left rotation amounts for the Rho step.
 *
 * The Rho step rotates lane A[x,y] left by RHO_OFFSETS[y][x] bits.
 * Lane A[0,0] has offset 0 and is never rotated.
 *
 * The offsets are derived from the formula (t+1)(t+2)/2 mod 64, where t
 * is determined by the iterative sequence starting at (x,y) = (1,0) and
 * applying (x,y) <- (y, 2x+3y mod 5) for t = 0..23. See FIPS 202
 * Section 3.2.2 for the full derivation.
 *
 * Indexed as [y][x] to match the state layout convention.
 */
 static const int RHO_OFFSETS[5][5] = {
    { 0, 1, 62, 28, 27},
    {36, 44,  6, 55, 20},
    { 3, 10, 43, 25, 39},
    {41, 45, 15, 21,  8},
    {18,  2, 61, 56, 14}
};

/**
 * ROUND_CONSTANTS — Iota step round constants for rounds 12 through 23.
 *
 * The Iota step XORs a round-specific constant into lane A[0,0] to break
 * the symmetry that would otherwise make every round identical. Without
 * this, the permutation would be vulnerable to slide attacks.
 *
 * The full Keccak-f[1600] has 24 round constants (rounds 0-23). Since
 * Keccak-p[1600, 12] runs only rounds 12-23, only the last 12 constants
 * are needed here. They are derived from a linear feedback shift register
 * defined in FIPS 202 Section 3.2.5.
 *
 * The index into this array is the round counter rnd = 0..11, which
 * corresponds to actual Keccak round number (rnd + 12).
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

/* -------------------------------------------------------------------------
 * keccak_p — the permutation
 *
 * Applies 12 rounds of the Keccak round function to the 1600-bit state.
 * Each round consists of the five step mappings described below.
 * The state is modified in-place.
 * ------------------------------------------------------------------------- */
void keccak_p(uint64_t state[25])
{
    for(int rnd = 0; rnd < K12_ROUNDS; ++rnd)
    {
        /* -----------------------------------------------------------------
         * Step 1: Theta — column parity diffusion
         *
         * Theta provides diffusion by mixing each lane with the parity of
         * two entire columns of the state. Without Theta, changing one
         * bit would affect only a single lane through the entire round.
         *
         * Algorithm:
         *   C[x] = A[x,0] XOR A[x,1] XOR A[x,2] XOR A[x,3] XOR A[x,4]
         *           (column parity: XOR all 5 lanes in column x)
         *
         *   D[x] = C[x-1] XOR ROT(C[x+1], 1)
         *           (mix adjacent column parities with a 1-bit rotation)
         *
         *   A[x,y] ^= D[x]  for all x, y
         *           (apply the mixed parity to every lane in the state)
         *
         * The result is that every lane is influenced by 11 other lanes
         * (the entire column to its left, and the entire column to its
         * right shifted by 1 bit). This gives Theta an algebraic branch
         * number of 4, the maximum for a 5-element operation.
         * ----------------------------------------------------------------- */
        uint64_t C[5];
        uint64_t D[5];

        /* Compute column parities — C[x] = XOR of all lanes in column x */
        for (int x = 0; x < 5; ++x) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }

        /* Mix adjacent column parities. Indices are mod 5 (wrap-around). */
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ rot64(C[(x + 1) % 5], 1);
        }

        /* XOR the mixed parity into every lane */
        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 5; ++x) {
                state[x + 5 * y] ^= D[x];
            }
        }

        /* -----------------------------------------------------------------
         * Steps 2 & 3: Rho and Pi — combined in one pass
         *
         * Rho rotates the bits within each lane by a fixed, lane-specific
         * offset. This provides intra-lane diffusion: a bit at position z
         * in lane A[x,y] moves to position (z + offset[x,y]) mod 64.
         * Lane A[0,0] is never rotated (offset 0).
         *
         * Pi rearranges the 25 lanes among the 25 positions in the state.
         * The new position of lane A[x,y] is A[y, 2x+3y mod 5].
         * Equivalently, the lane that lands at position A[x,y] comes from
         * A[x_old, y_old] = A[(x + 3y) mod 5, x].
         *
         * Rho and Pi are combined into a single loop to avoid an extra
         * 200-byte temporary. For each output position (x,y):
         *   1. Compute the source position (x_old, y_old) via Pi's inverse.
         *   2. Look up the Rho rotation offset for that source lane.
         *   3. Write rot64(state[source], offset) into next_state[dest].
         *
         * Together, Rho and Pi ensure that bits from different positions
         * and different bit-offsets within lanes are mixed across the state,
         * providing inter-lane diffusion to complement Theta's column mixing.
         * ----------------------------------------------------------------- */
        uint64_t next_state[25];
        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 5; ++x) {
                /* Pi inverse: the lane arriving at (x,y) came from (x_old, y_old) */
                int x_old = (x + 3 * y) % 5;
                int y_old = x;
                int shift = RHO_OFFSETS[y_old][x_old];

                next_state[x + 5 * y] = rot64(state[x_old + 5 * y_old], shift);
            }
        }

        /* -----------------------------------------------------------------
         * Step 4: Chi — non-linear row mixing
         *
         * Chi is the only non-linear step and is the source of Keccak's
         * resistance to linear cryptanalysis. It operates independently on
         * each row of 5 lanes, applying the bitwise function:
         *
         *   A[x,y] = A[x,y] XOR ((NOT A[x+1,y]) AND A[x+2,y])
         *
         * This is a 5-bit S-box applied in parallel across all 64 bit
         * positions of every row. The algebraic degree is 2 (quadratic),
         * which is the maximum degree achievable with a single AND gate.
         *
         * The temporary array T[] holds the unmodified row values so that
         * each output lane reads from the original inputs, not from
         * already-updated neighbours within the same row.
         * ----------------------------------------------------------------- */
        for (int y = 0; y < 5; ++y) {
            /* Snapshot the current row before any lane is overwritten */
            uint64_t T[5];
            for (int x = 0; x < 5; ++x) {
                T[x] = next_state[x + 5 * y];
            }
            /* Apply the Chi non-linearity to each lane in the row */
            for (int x = 0; x < 5; ++x) {
                state[x + 5 * y] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
            }
        }

        /* -----------------------------------------------------------------
         * Step 5: Iota — round constant injection
         *
         * Iota XORs a round-specific constant into lane A[0,0] (the single
         * lane at position state[0]). All other lanes are unchanged.
         *
         * Purpose: without Iota, every round would apply an identical
         * transformation, making the permutation vulnerable to slide attacks
         * (where an adversary can shift the round index by a constant and
         * find related inputs that produce related outputs). The round
         * constant breaks this round-to-round symmetry.
         *
         * Only lane A[0,0] is modified to keep the step cheap while still
         * breaking the symmetry. The constants are carefully chosen so that
         * each round is distinct and no constant is a simple transformation
         * of another.
         * ----------------------------------------------------------------- */
        state[0] ^= ROUND_CONSTANTS[rnd];
    }
}