#include "k12_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

#define K12_GPU_DOMAIN_LEAF    0x0Bu
#define K12_GPU_DOMAIN_ROOT    0x06u
#define K12_GPU_DOMAIN_SINGLE    0x07u
#define K12_GPU_HOP_MARKER_0     0x03u
#define K12_GPU_KECCAK_ROUNDS    12
#define K12_GPU_TURBOSHAKE_RATE  168

/* =============================================================================
 * Host
 * ============================================================================= */

namespace {

size_t max_leaves_for_jobs(const size_t* message_lengths,
                           const size_t* custom_lengths,
                           size_t count)
{
    size_t max_leaves = 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t S_len = message_lengths[i] + custom_lengths[i] + 9;
        max_leaves = std::max(max_leaves, k12_gpu_leaf_count(S_len));
    }
    return max_leaves;
}

} /* anonymous namespace */

void K12GpuItemBatch::upload(const uint8_t* const* items,
                             const size_t* lengths,
                             size_t count)
{
    free();
    if (count == 0) return;

    std::vector<size_t> offsets(count);
    size_t total_bytes = 0;
    for (size_t i = 0; i < count; ++i) {
        offsets[i] = total_bytes;
        total_bytes += lengths[i];
    }

    if (total_bytes > 0) {
        cudaMalloc(&d_data_, total_bytes);
    }
    cudaMalloc(&d_offsets_, count * sizeof(size_t));
    cudaMalloc(&d_lengths_, count * sizeof(size_t));
    cudaMemcpy(d_offsets_, offsets.data(), count * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths_, lengths, count * sizeof(size_t), cudaMemcpyHostToDevice);

    if (total_bytes > 0) {
        std::vector<uint8_t> staging(total_bytes);
        for (size_t i = 0; i < count; ++i) {
            if (lengths[i] == 0) continue;
            std::memcpy(staging.data() + offsets[i], items[i], lengths[i]);
        }
        cudaMemcpy(d_data_, staging.data(), total_bytes, cudaMemcpyHostToDevice);
    }

    count_ = count;
}

void K12GpuItemBatch::free()
{
    if (d_data_) cudaFree(d_data_);
    if (d_offsets_) cudaFree(d_offsets_);
    if (d_lengths_) cudaFree(d_lengths_);
    d_data_ = nullptr;
    d_offsets_ = nullptr;
    d_lengths_ = nullptr;
    count_ = 0;
}

void K12GpuBatch::upload(const uint8_t* const* messages,
                         const size_t* message_lengths,
                         const uint8_t* const* custom,
                         const size_t* custom_lengths,
                         const size_t* output_lengths,
                         size_t count)
{
    free();
    if (count == 0) return;

    messages_.upload(messages, message_lengths, count);
    customizations_.upload(custom, custom_lengths, count);

    output_offsets_.resize(count);
    size_t out_off = 0;
    for (size_t i = 0; i < count; ++i) {
        output_offsets_[i] = out_off;
        out_off += output_lengths[i];
    }

    total_output_bytes_ = out_off;
    max_leaves_per_job_ = max_leaves_for_jobs(message_lengths, custom_lengths, count);

    cudaMalloc(&d_output_, total_output_bytes_);
    cudaMalloc(&d_output_offsets_, count * sizeof(size_t));
    cudaMalloc(&d_output_lengths_, count * sizeof(size_t));
    cudaMemcpy(d_output_offsets_, output_offsets_.data(),
               count * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_lengths_, output_lengths,
               count * sizeof(size_t), cudaMemcpyHostToDevice);

    if (max_leaves_per_job_ > 0) {
        cudaMalloc(&d_leaf_cvs_, count * max_leaves_per_job_ * K12_GPU_CV_LEN);
    }
}

void K12GpuBatch::upload(const uint8_t* const* messages,
                         const size_t* message_lengths,
                         const uint8_t* const* custom,
                         const size_t* custom_lengths,
                         size_t count,
                         size_t output_len)
{
    if (count == 0) return;
    std::vector<size_t> lens(count, output_len);
    upload(messages, message_lengths, custom, custom_lengths, lens.data(), count);
}

void K12GpuBatch::free()
{
    messages_.free();
    customizations_.free();
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    if (d_leaf_cvs_) { cudaFree(d_leaf_cvs_); d_leaf_cvs_ = nullptr; }
    if (d_output_offsets_) { cudaFree(d_output_offsets_); d_output_offsets_ = nullptr; }
    if (d_output_lengths_) { cudaFree(d_output_lengths_); d_output_lengths_ = nullptr; }
    output_offsets_.clear();
    total_output_bytes_ = max_leaves_per_job_ = 0;
}

/* =============================================================================
 * Device: TurboSHAKE128 + Keccak-p[1600,12]  (matches CPU/keccak_p.cpp)
 * ============================================================================= */

__device__ __constant__ int k12_d_rho[5][5] = {
    { 0, 1, 62, 28, 27},
    {36, 44,  6, 55, 20},
    { 3, 10, 43, 25, 39},
    {41, 45, 15, 21,  8},
    {18,  2, 61, 56, 14}
};

__device__ __constant__ uint64_t k12_d_rc[12] = {
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ __forceinline__ uint64_t k12_gpu_rot64(uint64_t x, int n)
{
    n &= 63;
    return (x << n) | (x >> (64 - n));
}

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

struct K12GpuTurboSponge {
    uint64_t state[25];
    size_t   position;
};

__device__ inline void k12_gpu_ts_init(K12GpuTurboSponge* s)
{
    for (int i = 0; i < 25; ++i) s->state[i] = 0;
    s->position = 0;
}

/* XOR len bytes into the rate window; permute when full.
 * Uses uint64 XOR when state position and data pointer are 8-byte aligned. */
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

__device__ inline void k12_gpu_ts_pad(K12GpuTurboSponge* s, uint8_t domain)
{
    uint8_t* sb = reinterpret_cast<uint8_t*>(s->state);
    sb[s->position] ^= domain;
    sb[K12_GPU_TURBOSHAKE_RATE - 1] ^= 0x80u;
    k12_gpu_keccak_p(s->state);
    s->position = 0;
}

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

/* =============================================================================
 * Device: K12 kernel  (1 block = 1 message, blockIdx.x = job index)
 * ============================================================================= */

__device__ inline size_t k12_dev_length_encode(uint64_t x, uint8_t buf[9])
{
    if (x == 0) { buf[0] = 0; return 1; }
    uint8_t tmp[8];
    int n = 0;
    for (uint64_t v = x; v > 0; v >>= 8) tmp[n++] = static_cast<uint8_t>(v & 0xFFu);
    for (int i = 0; i < n; ++i) buf[i] = tmp[n - 1 - i];
    buf[n] = static_cast<uint8_t>(n);
    return static_cast<size_t>(n + 1);
}

__device__ inline size_t k12_dev_chunk_len(size_t S_len, size_t idx)
{
    const size_t start = idx * K12_GPU_CHUNK_SIZE;
    if (start >= S_len) return 0;
    size_t end = start + K12_GPU_CHUNK_SIZE;
    if (end > S_len) end = S_len;
    return end - start;
}

__device__ inline void k12_dev_absorb_stream_range(K12GpuTurboSponge* s,
                                                   size_t S_off,
                                                   size_t len,
                                                   size_t M_len,
                                                   const uint8_t* M,
                                                   size_t C_len,
                                                   const uint8_t* C,
                                                   const uint8_t* c_enc)
{
    const size_t end = S_off + len;
    while (S_off < end) {
        if (S_off < M_len) {
            size_t take = M_len - S_off;
            if (take > end - S_off) take = end - S_off;
            k12_gpu_ts_absorb(s, M + S_off, take);
            S_off += take;
        } else if (S_off < M_len + C_len) {
            const size_t pos = S_off - M_len;
            size_t take = C_len - pos;
            if (take > end - S_off) take = end - S_off;
            k12_gpu_ts_absorb(s, C + pos, take);
            S_off += take;
        } else {
            const size_t pos = S_off - M_len - C_len;
            const size_t take = end - S_off;
            k12_gpu_ts_absorb(s, c_enc + pos, take);
            S_off += take;
        }
    }
}

__device__ inline void k12_dev_absorb_chunk(K12GpuTurboSponge* s,
                                            size_t chunk_idx,
                                            size_t S_len,
                                            size_t M_len,
                                            const uint8_t* M,
                                            size_t C_len,
                                            const uint8_t* C,
                                            const uint8_t* c_enc)
{
    const size_t S_off = chunk_idx * K12_GPU_CHUNK_SIZE;
    k12_dev_absorb_stream_range(s, S_off, k12_dev_chunk_len(S_len, chunk_idx),
                                M_len, M, C_len, C, c_enc);
}

__global__ __launch_bounds__(K12_GPU_BLOCK_THREADS, K12_GPU_BLOCKS_PER_SM)
void k12_gpu_hash_kernel(const uint8_t* d_message_data,
                         const size_t* d_message_offsets,
                         const size_t* d_message_lengths,
                         const uint8_t* d_custom_data,
                         const size_t* d_custom_offsets,
                         const size_t* d_custom_lengths,
                         uint8_t* d_output,
                         const size_t* d_output_offsets,
                         const size_t* d_output_lengths,
                         size_t job_count,
                         uint8_t* d_leaf_cvs,
                         size_t max_leaves_per_job)
{
    const size_t job = blockIdx.x;
    if (job >= job_count) return;

    const uint8_t* M = d_message_data ? d_message_data + d_message_offsets[job] : nullptr;
    const size_t M_len = d_message_lengths[job];
    const uint8_t* C = d_custom_data ? d_custom_data + d_custom_offsets[job] : nullptr;
    const size_t C_len = d_custom_lengths[job];
    uint8_t* output = d_output + d_output_offsets[job];
    const size_t job_output_len = d_output_lengths[job];
    uint8_t* leaf_cvs = d_leaf_cvs + job * max_leaves_per_job * K12_GPU_CV_LEN;

    __shared__ uint8_t c_enc[9];
    __shared__ size_t  c_enc_len, S_len, chunk_count;

    if (threadIdx.x == 0) {
        c_enc_len = k12_dev_length_encode(static_cast<uint64_t>(C_len), c_enc);
        S_len = M_len + C_len + c_enc_len;
        chunk_count = (S_len + K12_GPU_CHUNK_SIZE - 1) / K12_GPU_CHUNK_SIZE;
    }
    __syncthreads();

    if (S_len <= K12_GPU_CHUNK_SIZE) {
        if (threadIdx.x == 0) {
            K12GpuTurboSponge s;
            k12_gpu_ts_init(&s);
            k12_dev_absorb_stream_range(&s, 0, S_len, M_len, M, C_len, C, c_enc);
            k12_gpu_ts_pad(&s, K12_GPU_DOMAIN_SINGLE);
            k12_gpu_ts_squeeze(&s, output, job_output_len);
        }
        return;
    }

    for (size_t wave = 1; wave < chunk_count; wave += K12_GPU_BLOCK_THREADS) {
        const size_t rem = chunk_count - wave;
        const size_t wc = rem < K12_GPU_BLOCK_THREADS ? rem : K12_GPU_BLOCK_THREADS;

        if (threadIdx.x < wc) {
            const size_t chunk_idx = wave + threadIdx.x;
            K12GpuTurboSponge s;
            k12_gpu_ts_init(&s);
            k12_dev_absorb_chunk(&s, chunk_idx, S_len, M_len, M, C_len, C, c_enc);
            k12_gpu_ts_pad(&s, K12_GPU_DOMAIN_LEAF);
            uint8_t cv[K12_GPU_CV_LEN];
            k12_gpu_ts_squeeze(&s, cv, K12_GPU_CV_LEN);
            std::memcpy(leaf_cvs + (chunk_idx - 1) * K12_GPU_CV_LEN, cv, K12_GPU_CV_LEN);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        K12GpuTurboSponge root;
        k12_gpu_ts_init(&root);
        k12_dev_absorb_chunk(&root, 0, S_len, M_len, M, C_len, C, c_enc);

        const uint8_t hop[8] = {K12_GPU_HOP_MARKER_0,0,0,0,0,0,0,0};
        k12_gpu_ts_absorb(&root, hop, 8);

        if (chunk_count > 1) {
            k12_gpu_ts_absorb(&root, leaf_cvs, (chunk_count - 1) * K12_GPU_CV_LEN);
        }

        uint8_t n_enc[9];
        const size_t n_enc_len =
            k12_dev_length_encode(static_cast<uint64_t>(chunk_count - 1), n_enc);
        k12_gpu_ts_absorb(&root, n_enc, n_enc_len);

        const uint8_t term[2] = {0xFFu, 0xFFu};
        k12_gpu_ts_absorb(&root, term, 2);
        k12_gpu_ts_pad(&root, K12_GPU_DOMAIN_ROOT);
        k12_gpu_ts_squeeze(&root, output, job_output_len);
    }
}

void k12_gpu_hash_batch(const K12GpuBatch& batch)
{
    const size_t jobs = batch.job_count();
    if (jobs == 0) return;

    k12_gpu_hash_kernel<<<static_cast<unsigned>(jobs), K12_GPU_BLOCK_THREADS>>>(
        batch.messages_.data(),
        batch.messages_.offsets(),
        batch.messages_.lengths(),
        batch.customizations_.data(),
        batch.customizations_.offsets(),
        batch.customizations_.lengths(),
        batch.d_output_,
        batch.d_output_offsets_,
        batch.d_output_lengths_,
        jobs,
        batch.d_leaf_cvs_,
        batch.max_leaves_per_job_);
    cudaDeviceSynchronize();
}
