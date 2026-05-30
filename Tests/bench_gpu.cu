#include "bench_common.h"

#include "../GPU/k12_gpu.h"

#include <cuda_runtime.h>

#include <vector>

static int run_case(size_t case_index, const BenchCase& c)
{
    bench_print_case(case_index, c);

    uint8_t* M = bench_make_pattern(c.message_bytes);
    if (!M) {
        std::fprintf(stderr, "malloc failed for %zu bytes\n", c.message_bytes);
        return 1;
    }

    std::vector<const uint8_t*> messages(c.num_messages, M);
    std::vector<const uint8_t*> customs(c.num_messages, nullptr);
    std::vector<size_t> message_lengths(c.num_messages, c.message_bytes);
    std::vector<size_t> custom_lengths(c.num_messages, 0);

    K12GpuBatch batch;

    const auto upload_t0 = std::chrono::steady_clock::now();
    batch.upload(messages.data(), message_lengths.data(),
               customs.data(), custom_lengths.data(),
               c.num_messages, BENCH_OUTPUT_LEN);
    const double upload_ms = bench_ms(std::chrono::steady_clock::now() - upload_t0);

    const auto hash_t0 = std::chrono::steady_clock::now();
    k12_gpu_hash_batch(batch);
    const double hash_ms = bench_ms(std::chrono::steady_clock::now() - hash_t0);

    std::vector<uint8_t> host_out(batch.total_output_bytes());
    const auto copy_t0 = std::chrono::steady_clock::now();
    cudaMemcpy(host_out.data(), batch.device_output(),
               batch.total_output_bytes(), cudaMemcpyDeviceToHost);
    const double copy_ms = bench_ms(std::chrono::steady_clock::now() - copy_t0);

    const double total_ms = upload_ms + hash_ms + copy_ms;
    const double hash_per_msg_ms =
        (c.num_messages > 0) ? (hash_ms / static_cast<double>(c.num_messages)) : hash_ms;

    std::printf("  upload:           %.3f ms\n", upload_ms);
    std::printf("  hash:             %.3f ms\n", hash_ms);
    std::printf("  hash (per msg):   %.3f ms\n", hash_per_msg_ms);
    std::printf("  copy:             %.3f ms\n", copy_ms);
    std::printf("  total:            %.3f ms\n", total_ms);
    std::printf("  digest[0..3]:     %02X %02X %02X %02X\n",
                host_out[0], host_out[1], host_out[2], host_out[3]);

    bench_metric_line("gpu", case_index, "upload", upload_ms);
    bench_metric_line("gpu", case_index, "hash", hash_ms);
    bench_metric_line("gpu", case_index, "copy", copy_ms);
    bench_metric_line("gpu", case_index, "total", total_ms);

    std::free(M);
    return 0;
}

int main(void)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::printf("=== KangarooTwelve timing benchmarks (GPU: %s) ===\n", prop.name);
    } else {
        std::printf("=== KangarooTwelve timing benchmarks (GPU) ===\n");
    }
    std::printf("  Message: ptn(n), customization empty, output %zu bytes\n",
                BENCH_OUTPUT_LEN);
    std::printf("  Cold start: first k12_gpu_hash_batch() immediately after upload\n");

    const size_t limit = bench_case_limit();
    if (limit < BENCH_CASE_COUNT) {
        std::printf("  Skipping case %zu (50 MB x50). Set BENCH_ALL=1 to include.\n",
                    BENCH_CASE_COUNT - 1);
    }

    int err = 0;
    for (size_t i = 0; i < limit; ++i) {
        std::printf("Starting case %zu...\n", i);
        fflush(stdout);
        if (run_case(i, BENCH_CASES[i]) != 0) {
            err = 1;
        }
    }
    return err;
}
