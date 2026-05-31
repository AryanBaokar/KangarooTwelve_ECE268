#pragma once

#include <cstddef>
#include <cstdint>

#define K12_GPU_KECCAK_ROUNDS   12
#define K12_GPU_TURBOSHAKE_RATE 168

/* Rho step rotation offsets (5x5 lane grid). */
__device__ __constant__ int k12_d_rho[5][5] = {
    { 0, 1, 62, 28, 27},
    {36, 44,  6, 55, 20},
    { 3, 10, 43, 25, 39},
    {41, 45, 15, 21,  8},
    {18,  2, 61, 56, 14}
};

/* Iota round constants for Keccak-p[1600, 12] (rounds 12-23). */
__device__ __constant__ uint64_t k12_d_rc[12] = {
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

/* TurboSHAKE128 sponge state (Keccak-f[1600] with 168-byte rate). */
struct K12GpuTurboSponge {
    uint64_t state[25];
    size_t   position;
};

/* 64-bit left rotation used by the Rho step. */
__device__ __forceinline__ uint64_t k12_gpu_rot64(uint64_t x, int n)
{
    n &= 63;
    return (x << n) | (x >> (64 - n));
}

/* Keccak-p[1600, 12]: 12 rounds of Theta, Rho/Pi, Chi, and Iota on 25 lanes. */
__device__ void k12_gpu_keccak_p(uint64_t state[25])
{
    for (int rnd = 0; rnd < K12_GPU_KECCAK_ROUNDS; ++rnd) {
        uint64_t C[5];
        uint64_t D[5];

        for (int x = 0; x < 5; ++x) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ k12_gpu_rot64(C[(x + 1) % 5], 1);
        }
        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 5; ++x) {
                state[x + 5 * y] ^= D[x];
            }
        }

        uint64_t next_state[25];
        for (int y = 0; y < 5; ++y) {
            for (int x = 0; x < 5; ++x) {
                const int x_old = (x + 3 * y) % 5;
                const int y_old = x;
                const int shift = k12_d_rho[y_old][x_old];
                next_state[x + 5 * y] = k12_gpu_rot64(state[x_old + 5 * y_old], shift);
            }
        }

        for (int y = 0; y < 5; ++y) {
            uint64_t T[5];
            for (int x = 0; x < 5; ++x) {
                T[x] = next_state[x + 5 * y];
            }
            for (int x = 0; x < 5; ++x) {
                state[x + 5 * y] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5]);
            }
        }

        state[0] ^= k12_d_rc[rnd];
    }
}

/* Zero sponge state and reset the rate cursor. */
__device__ inline void k12_gpu_ts_init(K12GpuTurboSponge* s)
{
    for (int i = 0; i < 25; ++i) s->state[i] = 0;
    s->position = 0;
}

/* XOR input into the rate; permute when the 168-byte window is full. */
__device__ inline void k12_gpu_ts_absorb(K12GpuTurboSponge* s,
                                         const uint8_t* data,
                                         size_t len)
{
    uint8_t* sb = reinterpret_cast<uint8_t*>(s->state);
    size_t off = 0;

    while (off < len) {
        if (s->position == K12_GPU_TURBOSHAKE_RATE) {
            k12_gpu_keccak_p(s->state);
            s->position = 0;
        }

        size_t space = K12_GPU_TURBOSHAKE_RATE - s->position;
        size_t take = len - off;
        if (take > space) {
            take = space;
        }

        const uintptr_t data_addr = reinterpret_cast<uintptr_t>(data + off);
        if (s->position % 8 == 0 && (data_addr % 8) == 0 && take >= 8) {
            const size_t qtake = take & ~size_t(7);
            uint64_t* st8 = reinterpret_cast<uint64_t*>(sb + s->position);
            const uint64_t* in8 = reinterpret_cast<const uint64_t*>(data + off);
            const size_t nq = qtake / 8;
            for (size_t q = 0; q < nq; ++q) {
                st8[q] ^= in8[q];
            }
            s->position += qtake;
            off += qtake;
            take -= qtake;
        }

        for (size_t i = 0; i < take; ++i) {
            sb[s->position++] ^= data[off++];
        }

        if (s->position == K12_GPU_TURBOSHAKE_RATE) {
            k12_gpu_keccak_p(s->state);
            s->position = 0;
        }
    }
}

/* Apply domain byte and padding bit, then permute (sponge squeeze setup). */
__device__ inline void k12_gpu_ts_pad(K12GpuTurboSponge* s, uint8_t domain)
{
    uint8_t* sb = reinterpret_cast<uint8_t*>(s->state);
    sb[s->position] ^= domain;
    sb[K12_GPU_TURBOSHAKE_RATE - 1] ^= 0x80u;
    k12_gpu_keccak_p(s->state);
    s->position = 0;
}

/* Read len bytes from the rate, permuting as needed between blocks. */
__device__ inline void k12_gpu_ts_squeeze(K12GpuTurboSponge* s, uint8_t* out, size_t len)
{
    uint8_t* sb = reinterpret_cast<uint8_t*>(s->state);
    size_t off = 0;
    while (off < len) {
        if (s->position == K12_GPU_TURBOSHAKE_RATE) {
            k12_gpu_keccak_p(s->state);
            s->position = 0;
        }
        size_t take = len - off;
        if (take > K12_GPU_TURBOSHAKE_RATE - s->position) {
            take = K12_GPU_TURBOSHAKE_RATE - s->position;
        }
        for (size_t i = 0; i < take; ++i) out[off + i] = sb[s->position++];
        off += take;
    }
}
