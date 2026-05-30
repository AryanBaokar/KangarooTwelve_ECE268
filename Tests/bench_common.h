#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct BenchCase {
    const char* title;
    size_t      message_bytes;
    size_t      num_messages;
};

inline constexpr BenchCase BENCH_CASES[] = {
    { "1 KB x1 message",    1024u,                   1u },
    { "8 KB x1 message",    8192u,                   1u },
    { "24 KB x1 message",   24u * 1024u,             1u },
    { "48 KB x1 message",   48u * 1024u,             1u },
    { "1 MB x1 message",    1024u * 1024u,           1u },
    { "1 MB x10 messages",  1024u * 1024u,          10u },
    { "50 MB x50 messages", 50u * 1024u * 1024u,    50u },
};

inline constexpr size_t BENCH_CASE_COUNT = sizeof(BENCH_CASES) / sizeof(BENCH_CASES[0]);
/* Cases 0..5 by default; case 6 (50x50 MB) only with BENCH_ALL=1 */
inline constexpr size_t BENCH_QUICK_CASE_COUNT = BENCH_CASE_COUNT - 1u;
inline constexpr size_t BENCH_OUTPUT_LEN = 32u;

inline size_t bench_case_limit(void)
{
    const char* all = std::getenv("BENCH_ALL");
    if (all && all[0] == '1' && all[1] == '\0') {
        return BENCH_CASE_COUNT;
    }
    return BENCH_QUICK_CASE_COUNT;
}

inline uint8_t* bench_make_pattern(size_t n)
{
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(n ? n : 1u));
    if (!buf) {
        return nullptr;
    }
    for (size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<uint8_t>(i % 251u);
    }
    return buf;
}

inline double bench_ms(std::chrono::steady_clock::duration d)
{
    return std::chrono::duration<double, std::milli>(d).count();
}

inline void bench_print_case(size_t case_index, const BenchCase& c)
{
    std::printf("\n--- Case %zu: %s ---\n", case_index, c.title);
    std::printf("  message_bytes=%zu num_messages=%zu\n",
                c.message_bytes, c.num_messages);
}

inline void bench_metric_line(const char* backend, size_t case_index,
                             const char* metric, double ms)
{
    std::printf("BENCH_METRIC backend=%s case=%zu metric=%s ms=%.6f\n",
                backend, case_index, metric, ms);
}
