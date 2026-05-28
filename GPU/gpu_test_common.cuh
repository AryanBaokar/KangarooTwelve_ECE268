#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

inline void gpu_print_duration(const char* label, std::chrono::steady_clock::duration elapsed)
{
    const double us = std::chrono::duration<double, std::micro>(elapsed).count();
    if (us < 1000.0) {
        std::printf("  %s: %.3f us\n", label, us);
    } else if (us < 1000000.0) {
        std::printf("  %s: %.3f ms\n", label, us / 1000.0);
    } else {
        std::printf("  %s: %.3f s\n", label, us / 1000000.0);
    }
}

inline void gpu_print_throughput(const char* label,
                                 size_t message_bytes,
                                 size_t num_messages,
                                 std::chrono::steady_clock::duration elapsed)
{
    const double s = std::chrono::duration<double>(elapsed).count();
    if (s <= 0.0) return;
    const double total_mib =
        (static_cast<double>(message_bytes) * static_cast<double>(num_messages))
        / (1024.0 * 1024.0);
    std::printf("  %s: %.2f MiB/s (%.0f MiB total)\n", label, total_mib / s, total_mib);
}

inline uint8_t* gpu_make_pattern(size_t n)
{
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(n ? n : 1));
    for (size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<uint8_t>(i % 251);
    }
    return buf;
}
