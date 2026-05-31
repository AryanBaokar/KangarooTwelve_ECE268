#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

/* KangarooTwelve GPU batch hasher - 1 CUDA block per message, 128 threads/block.
 *
 * Build: nvcc -std=c++17 -O2 k12_gpu.cu your_test.cu -o your_test */

#define K12_GPU_CHUNK_SIZE    8192u
#define K12_GPU_CV_LEN        32u
#define K12_GPU_BLOCK_THREADS 128u
#define K12_GPU_BLOCKS_PER_SM 1u

/* Number of leaf CV slots for a serialized input of length S_len. */
inline size_t k12_gpu_leaf_count(size_t S_len)
{
    const size_t chunks = (S_len + K12_GPU_CHUNK_SIZE - 1) / K12_GPU_CHUNK_SIZE;
    return (chunks > 0) ? (chunks - 1u) : 0u;
}

/* Packed device buffers for a list of variable-length byte blobs. */
class K12GpuItemBatch {
public:
    K12GpuItemBatch() = default;
    ~K12GpuItemBatch() { free(); }
    K12GpuItemBatch(const K12GpuItemBatch&) = delete;
    K12GpuItemBatch& operator=(const K12GpuItemBatch&) = delete;

    /* Concatenate items on the host and copy offsets/lengths/data to the GPU. */
    void upload(const uint8_t* const* items, const size_t* lengths, size_t count);
    void free();

    size_t count() const { return count_; }
    uint8_t* data() const { return d_data_; }
    const size_t* offsets() const { return d_offsets_; }
    const size_t* lengths() const { return d_lengths_; }

private:
    uint8_t* d_data_ = nullptr;
    size_t*  d_offsets_ = nullptr;
    size_t*  d_lengths_ = nullptr;
    size_t   count_ = 0;
};

/* One KangarooTwelve job per batch entry (message, customization, output length). */
class K12GpuBatch {
public:
    K12GpuBatch() = default;
    ~K12GpuBatch() { free(); }
    K12GpuBatch(const K12GpuBatch&) = delete;
    K12GpuBatch& operator=(const K12GpuBatch&) = delete;

    void upload(const uint8_t* const* messages,
                const size_t* message_lengths,
                const uint8_t* const* custom,
                const size_t* custom_lengths,
                const size_t* output_lengths,
                size_t count);

    /* Same as above when every job requests the same digest length. */
    void upload(const uint8_t* const* messages,
                const size_t* message_lengths,
                const uint8_t* const* custom,
                const size_t* custom_lengths,
                size_t count,
                size_t output_len);

    void free();

    size_t job_count() const { return messages_.count(); }
    size_t total_output_bytes() const { return total_output_bytes_; }
    uint8_t* device_output() const { return d_output_; }

private:
    friend void k12_gpu_hash_batch(const K12GpuBatch& batch);

    K12GpuItemBatch messages_;
    K12GpuItemBatch customizations_;
    uint8_t* d_output_ = nullptr;
    uint8_t* d_leaf_cvs_ = nullptr;
    size_t*  d_output_offsets_ = nullptr;
    size_t*  d_output_lengths_ = nullptr;
    std::vector<size_t> output_offsets_;
    size_t total_output_bytes_ = 0;
    size_t max_leaves_per_job_ = 0;
};

/* Launch one K12 kernel block per uploaded job and wait for completion. */
void k12_gpu_hash_batch(const K12GpuBatch& batch);
