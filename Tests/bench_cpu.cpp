#include "bench_common.h"

#include "../CPU/k12.h"

#include <vector>

static int run_case(size_t case_index, const BenchCase& c)
{
    bench_print_case(case_index, c);

    uint8_t* M = bench_make_pattern(c.message_bytes);
    if (!M) {
        std::fprintf(stderr, "malloc failed for %zu bytes\n", c.message_bytes);
        return 1;
    }

    std::vector<uint8_t> out(BENCH_OUTPUT_LEN);

    const auto hash_t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < c.num_messages; ++i) {
        kangaroo_twelve(M, c.message_bytes, nullptr, 0, out.data(), BENCH_OUTPUT_LEN);
    }
    const double hash_ms = bench_ms(std::chrono::steady_clock::now() - hash_t0);

    const double hash_per_msg_ms =
        (c.num_messages > 0) ? (hash_ms / static_cast<double>(c.num_messages)) : hash_ms;

    std::printf("  hash (total):     %.3f ms\n", hash_ms);
    std::printf("  hash (per msg):   %.3f ms\n", hash_per_msg_ms);
    std::printf("  digest[0..3]:     %02X %02X %02X %02X\n",
                out[0], out[1], out[2], out[3]);

    bench_metric_line("cpu", case_index, "hash", hash_ms);
    bench_metric_line("cpu", case_index, "total", hash_ms);

    std::free(M);
    return 0;
}

int main(void)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const size_t limit = bench_case_limit();
    std::printf("=== KangarooTwelve timing benchmarks (CPU) ===\n");
    std::printf("  Message: ptn(n), customization empty, output %zu bytes\n",
                BENCH_OUTPUT_LEN);
    std::printf("  Cold start: first kangaroo_twelve() per case (batch cases loop all jobs)\n");
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
