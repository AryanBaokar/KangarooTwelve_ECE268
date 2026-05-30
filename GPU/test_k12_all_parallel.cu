#include "k12_gpu.h"
#include "gpu_test_common.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* All RFC 9861 CPU test vectors in one parallel GPU batch (1 block = 1 vector).
 *
 * Build (from GPU/):
 *   nvcc -std=c++17 -O2 k12_gpu.cu test_k12_all_parallel.cu -o test_k12_all_parallel
 *
 * Run:
 *   ./test_k12_all_parallel
 */

/* Hex-dump bytes for failed vector comparison. */
static void print_hex(const char* label, const uint8_t* data, size_t len)
{
    std::printf("  %s: ", label);
    for (size_t i = 0; i < len; ++i) {
        std::printf("%02X ", data[i]);
        if ((i + 1) % 16 == 0) {
            std::printf("\n         ");
        }
    }
    std::printf("\n");
}

/* std::vector version of ptn(n): byte i is i % 251. */
static std::vector<uint8_t> make_pattern(size_t n)
{
    std::vector<uint8_t> buf(n ? n : 1);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<uint8_t>(i % 251);
    }
    return buf;
}

struct TestCase {
    const char* name;
    std::vector<uint8_t> message;
    std::vector<uint8_t> custom;
    size_t output_len;
    size_t check_offset;
    size_t check_len;
    const uint8_t* expected;
};

/* Populate all 14 RFC 9861 KangarooTwelve CPU reference vectors for GPU batch verify. */
static std::vector<TestCase> build_all_test_cases()
{
    static const uint8_t exp_empty_32[32] = {
        0x1A,0xC2,0xD4,0x50,0xFC,0x3B,0x42,0x05,0xD1,0x9D,0xA7,0xBF,0xCA,0x1B,0x37,0x51,
        0x3C,0x08,0x03,0x57,0x7A,0xC7,0x16,0x7F,0x06,0xFE,0x2C,0xE1,0xF0,0xEF,0x39,0xE5
    };
    static const uint8_t exp_empty_64[64] = {
        0x1A,0xC2,0xD4,0x50,0xFC,0x3B,0x42,0x05,0xD1,0x9D,0xA7,0xBF,0xCA,0x1B,0x37,0x51,
        0x3C,0x08,0x03,0x57,0x7A,0xC7,0x16,0x7F,0x06,0xFE,0x2C,0xE1,0xF0,0xEF,0x39,0xE5,
        0x42,0x69,0xC0,0x56,0xB8,0xC8,0x2E,0x48,0x27,0x60,0x38,0xB6,0xD2,0x92,0x96,0x6C,
        0xC0,0x7A,0x3D,0x46,0x45,0x27,0x2E,0x31,0xFF,0x38,0x50,0x81,0x39,0xEB,0x0A,0x71
    };
    static const uint8_t exp_empty_10032_last32[32] = {
        0xE8,0xDC,0x56,0x36,0x42,0xF7,0x22,0x8C,0x84,0x68,0x4C,0x89,0x84,0x05,0xD3,0xA8,
        0x34,0x79,0x91,0x58,0xC0,0x79,0xB1,0x28,0x80,0x27,0x7A,0x1D,0x28,0xE2,0xFF,0x6D
    };
    static const uint8_t exp_ptn1[32] = {
        0x2B,0xDA,0x92,0x45,0x0E,0x8B,0x14,0x7F,0x8A,0x7C,0xB6,0x29,0xE7,0x84,0xA0,0x58,
        0xEF,0xCA,0x7C,0xF7,0xD8,0x21,0x8E,0x02,0xD3,0x45,0xDF,0xAA,0x65,0x24,0x4A,0x1F
    };
    static const uint8_t exp_ptn17[32] = {
        0x6B,0xF7,0x5F,0xA2,0x23,0x91,0x98,0xDB,0x47,0x72,0xE3,0x64,0x78,0xF8,0xE1,0x9B,
        0x0F,0x37,0x12,0x05,0xF6,0xA9,0xA9,0x3A,0x27,0x3F,0x51,0xDF,0x37,0x12,0x28,0x88
    };
    static const uint8_t exp_ptn17sq[32] = {
        0x0C,0x31,0x5E,0xBC,0xDE,0xDB,0xF6,0x14,0x26,0xDE,0x7D,0xCF,0x8F,0xB7,0x25,0xD1,
        0xE7,0x46,0x75,0xD7,0xF5,0x32,0x7A,0x50,0x67,0xF3,0x67,0xB1,0x08,0xEC,0xB6,0x7C
    };
    static const uint8_t exp_ptn17cube[32] = {
        0xCB,0x55,0x2E,0x2E,0xC7,0x7D,0x99,0x10,0x70,0x1D,0x57,0x8B,0x45,0x7D,0xDF,0x77,
        0x2C,0x12,0xE3,0x22,0xE4,0xEE,0x7F,0xE4,0x17,0xF9,0x2C,0x75,0x8F,0x0D,0x59,0xD0
    };
    static const uint8_t exp_ptn17_4[32] = {
        0x87,0x01,0x04,0x5E,0x22,0x20,0x53,0x45,0xFF,0x4D,0xDA,0x05,0x55,0x5C,0xBB,0x5C,
        0x3A,0xF1,0xA7,0x71,0xC2,0xB8,0x9B,0xAE,0xF3,0x7D,0xB4,0x3D,0x99,0x98,0xB9,0xFE
    };
    static const uint8_t exp_ptn17_5[32] = {
        0x84,0x4D,0x61,0x09,0x33,0xB1,0xB9,0x96,0x3C,0xBD,0xEB,0x5A,0xE3,0xB6,0xB0,0x5C,
        0xC7,0xCB,0xD6,0x7C,0xEE,0xDF,0x88,0x3E,0xB6,0x78,0xA0,0xA8,0xE0,0x37,0x16,0x82
    };
    static const uint8_t exp_ptn17_6[32] = {
        0x3C,0x39,0x07,0x82,0xA8,0xA4,0xE8,0x9F,0xA6,0x36,0x7F,0x72,0xFE,0xAA,0xF1,0x32,
        0x55,0xC8,0xD9,0x58,0x78,0x48,0x1D,0x3C,0xD8,0xCE,0x85,0xF5,0x8E,0x88,0x0A,0xF8
    };
    static const uint8_t exp_custom_1[32] = {
        0xFA,0xB6,0x58,0xDB,0x63,0xE9,0x4A,0x24,0x61,0x88,0xBF,0x7A,0xF6,0x9A,0x13,0x30,
        0x45,0xF4,0x6E,0xE9,0x84,0xC5,0x6E,0x3C,0x33,0x28,0xCA,0xAF,0x1A,0xA1,0xA5,0x83
    };
    static const uint8_t exp_custom_41[32] = {
        0xD8,0x48,0xC5,0x06,0x8C,0xED,0x73,0x6F,0x44,0x62,0x15,0x9B,0x98,0x67,0xFD,0x4C,
        0x20,0xB8,0x08,0xAC,0xC3,0xD5,0xBC,0x48,0xE0,0xB0,0x6B,0xA0,0xA3,0x76,0x2E,0xC4
    };
    static const uint8_t exp_custom_41sq[32] = {
        0xC3,0x89,0xE5,0x00,0x9A,0xE5,0x71,0x20,0x85,0x4C,0x2E,0x8C,0x64,0x67,0x0A,0xC0,
        0x13,0x58,0xCF,0x4C,0x1B,0xAF,0x89,0x44,0x7A,0x72,0x42,0x34,0xDC,0x7C,0xED,0x74
    };
    static const uint8_t exp_custom_41cube[32] = {
        0x75,0xD2,0xF8,0x6A,0x2E,0x64,0x45,0x66,0x72,0x6B,0x4F,0xBC,0xFC,0x56,0x57,0xB9,
        0xDB,0xCF,0x07,0x0C,0x7B,0x0D,0xCA,0x06,0x45,0x0A,0xB2,0x91,0xD7,0x44,0x3B,0xCF
    };

    std::vector<TestCase> cases;
    cases.reserve(14);

    auto add = [&](const char* name,
                   std::vector<uint8_t> m,
                   std::vector<uint8_t> c,
                   size_t out_len,
                   size_t check_off,
                   size_t check_len,
                   const uint8_t* expected) {
        TestCase tc{};
        tc.name = name;
        tc.message = std::move(m);
        tc.custom = std::move(c);
        tc.output_len = out_len;
        tc.check_offset = check_off;
        tc.check_len = check_len;
        tc.expected = expected;
        cases.push_back(std::move(tc));
    };

    add("K12(M='', C='', L=32)", {}, {}, 32, 0, 32, exp_empty_32);
    add("K12(M='', C='', L=64)", {}, {}, 64, 0, 64, exp_empty_64);
    add("K12(M='', C='', L=10032) last 32B", {}, {}, 10032, 10032 - 32, 32, exp_empty_10032_last32);

    std::printf("Building test inputs (large patterns may take a moment)...\n");

    add("K12(M=ptn(1), C='', L=32)", make_pattern(1), {}, 32, 0, 32, exp_ptn1);
    add("K12(M=ptn(17), C='', L=32)", make_pattern(17), {}, 32, 0, 32, exp_ptn17);
    add("K12(M=ptn(17^2), C='', L=32)", make_pattern(17 * 17), {}, 32, 0, 32, exp_ptn17sq);
    add("K12(M=ptn(17^3), C='', L=32)", make_pattern(17 * 17 * 17), {}, 32, 0, 32, exp_ptn17cube);
    add("K12(M=ptn(17^4), C='', L=32)", make_pattern(17UL * 17 * 17 * 17), {}, 32, 0, 32, exp_ptn17_4);

    std::printf("  building ptn(17^5)...\n");
    add("K12(M=ptn(17^5), C='', L=32)", make_pattern(17UL * 17 * 17 * 17 * 17), {}, 32, 0, 32, exp_ptn17_5);

    std::printf("  building ptn(17^6)...\n");
    add("K12(M=ptn(17^6), C='', L=32)", make_pattern(17UL * 17 * 17 * 17 * 17 * 17), {}, 32, 0, 32, exp_ptn17_6);

    add("K12(M='', C=ptn(1), L=32)", {}, make_pattern(1), 32, 0, 32, exp_custom_1);
    add("K12(M=FF, C=ptn(41), L=32)", std::vector<uint8_t>{0xFF}, make_pattern(41), 32, 0, 32, exp_custom_41);
    add("K12(M=FF*3, C=ptn(41^2), L=32)",
        std::vector<uint8_t>{0xFF, 0xFF, 0xFF},
        make_pattern(41 * 41),
        32, 0, 32, exp_custom_41sq);
    add("K12(M=FF*7, C=ptn(41^3), L=32)",
        std::vector<uint8_t>(7, 0xFF),
        make_pattern(41 * 41 * 41),
        32, 0, 32, exp_custom_41cube);

    std::printf("  %zu test cases ready.\n\n", cases.size());
    return cases;
}

int main(void)
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        std::printf("=== KangarooTwelve parallel batch test (device: %s, CC %d.%d) ===\n\n",
                    prop.name, prop.major, prop.minor);
    } else {
        std::printf("=== KangarooTwelve parallel batch test ===\n\n");
    }

    const std::vector<TestCase> cases = build_all_test_cases();
    const size_t n = cases.size();

    std::vector<const uint8_t*> messages(n);
    std::vector<const uint8_t*> customs(n);
    std::vector<size_t> message_lengths(n);
    std::vector<size_t> custom_lengths(n);
    std::vector<size_t> output_lengths(n);

    for (size_t i = 0; i < n; ++i) {
        message_lengths[i] = cases[i].message.size();
        custom_lengths[i] = cases[i].custom.size();
        output_lengths[i] = cases[i].output_len;
        messages[i] = message_lengths[i] ? cases[i].message.data() : nullptr;
        customs[i] = custom_lengths[i] ? cases[i].custom.data() : nullptr;
    }

    size_t total_msg = 0, total_cust = 0, total_out = 0;
    for (size_t i = 0; i < n; ++i) {
        total_msg += message_lengths[i];
        total_cust += custom_lengths[i];
        total_out += output_lengths[i];
    }
    std::printf("Batch host input: messages %.2f MiB, custom %.2f KiB, output %.2f KiB\n",
                total_msg / (1024.0 * 1024.0),
                total_cust / 1024.0,
                total_out / 1024.0);

    K12GpuBatch batch;

    const auto upload_t0 = std::chrono::steady_clock::now();
    batch.upload(messages.data(), message_lengths.data(),
                 customs.data(), custom_lengths.data(),
                 output_lengths.data(), n);
    const auto upload_time = std::chrono::steady_clock::now() - upload_t0;

    std::printf("Launching %zu blocks (1 block per test vector, %u threads/block)\n",
                n, static_cast<unsigned>(K12_GPU_BLOCK_THREADS));

    const auto hash_t0 = std::chrono::steady_clock::now();
    k12_gpu_hash_batch(batch);
    const auto hash_time = std::chrono::steady_clock::now() - hash_t0;

    std::vector<uint8_t> host_out(batch.total_output_bytes());

    const auto copy_t0 = std::chrono::steady_clock::now();
    cudaMemcpy(host_out.data(), batch.device_output(),
               batch.total_output_bytes(), cudaMemcpyDeviceToHost);
    const auto download_time = std::chrono::steady_clock::now() - copy_t0;

    std::printf("\n=== Timing (all %zu hashes in one kernel launch) ===\n", n);
    gpu_print_duration("upload", upload_time);
    gpu_print_duration("hash  ", hash_time);
    gpu_print_duration("copy  ", download_time);
    gpu_print_duration("total ", upload_time + hash_time + download_time);

    int passed = 0;
    std::printf("\n=== Per-vector results ===\n\n");
    for (size_t i = 0; i < n; ++i) {
        size_t out_base = 0;
        for (size_t j = 0; j < i; ++j) {
            out_base += cases[j].output_len;
        }
        const uint8_t* got = host_out.data() + out_base + cases[i].check_offset;

        const bool ok = (std::memcmp(got, cases[i].expected, cases[i].check_len) == 0);
        if (ok) {
            ++passed;
            std::printf("[PASS] [%zu] %s\n", i, cases[i].name);
        } else {
            std::printf("[FAIL] [%zu] %s\n", i, cases[i].name);
            print_hex("expected", cases[i].expected, cases[i].check_len);
            print_hex("got     ", got, cases[i].check_len);
        }
    }

    std::printf("\n=== Results: %d / %zu passed ===\n", passed, n);
    return (passed == static_cast<int>(n)) ? 0 : 1;
}
