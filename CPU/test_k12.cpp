#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include "k12.h"

/* -------------------------------------------------------------------------
 * Test infrastructure
 * g++ -std=c++17 -O2 -Wall keccak_p.cpp turboshake128.cpp k12.cpp test_k12.cpp -o test_k12
 * ------------------------------------------------------------------------- */

static int tests_run    = 0;
static int tests_passed = 0;
static std::chrono::steady_clock::duration total_k12_time{};

static void print_hex(const char* label, const uint8_t* data, size_t len)
{
    printf("  %s: ", label);
    for (size_t i = 0; i < len; ++i) {
        printf("%02X ", data[i]);
        if ((i + 1) % 16 == 0) printf("\n         ");
    }
    printf("\n");
}

static void print_duration(const char* label, std::chrono::steady_clock::duration elapsed)
{
    const double us = std::chrono::duration<double, std::micro>(elapsed).count();
    if (us < 1000.0) {
        printf("  %s: %.3f us\n", label, us);
    } else if (us < 1000000.0) {
        printf("  %s: %.3f ms\n", label, us / 1000.0);
    } else {
        printf("  %s: %.3f s\n", label, us / 1000000.0);
    }
}

static void check(const char* name,
                  std::chrono::steady_clock::duration elapsed,
                  const uint8_t* got, const uint8_t* expected, size_t len)
{
    tests_run++;
    if (memcmp(got, expected, len) == 0) {
        tests_passed++;
        printf("[PASS] %s\n", name);
    } else {
        printf("[FAIL] %s\n", name);
        print_hex("expected", expected, len);
        print_hex("got     ", got,      len);
    }
    print_duration("k12 time", elapsed);
    total_k12_time += elapsed;
}

#define RUN_K12_AND_CHECK(name, out, expected, outlen, ...) \
    do { \
        const auto t0 = std::chrono::steady_clock::now(); \
        kangaroo_twelve(__VA_ARGS__); \
        const auto elapsed = std::chrono::steady_clock::now() - t0; \
        check(name, elapsed, out, expected, outlen); \
    } while (0)

/* -------------------------------------------------------------------------
 * Pattern generator: ptn(n) = 00 01 02 ... FA 00 01 ... (cycle of 251 bytes)
 * ------------------------------------------------------------------------- */
static uint8_t* make_pattern(size_t n)
{
    uint8_t* buf = (uint8_t*)malloc(n ? n : 1);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = (uint8_t)(i % 251);
    }
    return buf;
}

/* -------------------------------------------------------------------------
 * Official test vectors — draft-irtf-cfrg-kangarootwelve-07, Section 3
 * https://datatracker.ietf.org/doc/rfc9861/                       
 * All vectors use 32 output bytes unless noted.
 * ------------------------------------------------------------------------- */

/* KangarooTwelve(M=``, C=``, 32) */
static void test_empty_32(void)
{
    const uint8_t expected[32] = {
        0x1A,0xC2,0xD4,0x50,0xFC,0x3B,0x42,0x05,
        0xD1,0x9D,0xA7,0xBF,0xCA,0x1B,0x37,0x51,
        0x3C,0x08,0x03,0x57,0x7A,0xC7,0x16,0x7F,
        0x06,0xFE,0x2C,0xE1,0xF0,0xEF,0x39,0xE5
    };
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M='', C='', L=32)", out, expected, 32,
                      NULL, 0, NULL, 0, out, 32);
}

/* KangarooTwelve(M=``, C=``, 64) - first 32 bytes must match the 32-byte vector */
static void test_empty_64(void)
{
    const uint8_t expected[64] = {
        0x1A,0xC2,0xD4,0x50,0xFC,0x3B,0x42,0x05,
        0xD1,0x9D,0xA7,0xBF,0xCA,0x1B,0x37,0x51,
        0x3C,0x08,0x03,0x57,0x7A,0xC7,0x16,0x7F,
        0x06,0xFE,0x2C,0xE1,0xF0,0xEF,0x39,0xE5,
        0x42,0x69,0xC0,0x56,0xB8,0xC8,0x2E,0x48,
        0x27,0x60,0x38,0xB6,0xD2,0x92,0x96,0x6C,
        0xC0,0x7A,0x3D,0x46,0x45,0x27,0x2E,0x31,
        0xFF,0x38,0x50,0x81,0x39,0xEB,0x0A,0x71
    };
    uint8_t out[64];
    RUN_K12_AND_CHECK("K12(M='', C='', L=64)", out, expected, 64,
                      NULL, 0, NULL, 0, out, 64);
}

/* KangarooTwelve(M=``, C=``, 10032) - last 32 bytes */
static void test_empty_10032_last32(void)
{
    const uint8_t expected[32] = {
        0xE8,0xDC,0x56,0x36,0x42,0xF7,0x22,0x8C,
        0x84,0x68,0x4C,0x89,0x84,0x05,0xD3,0xA8,
        0x34,0x79,0x91,0x58,0xC0,0x79,0xB1,0x28,
        0x80,0x27,0x7A,0x1D,0x28,0xE2,0xFF,0x6D
    };
    uint8_t* out = (uint8_t*)malloc(10032);
    RUN_K12_AND_CHECK("K12(M='', C='', L=10032) last 32B", out + 10032 - 32, expected, 32,
                      NULL, 0, NULL, 0, out, 10032);
    free(out);
}

/* KangarooTwelve(M=ptn(1), C=``, 32) */
static void test_ptn1(void)
{
    const uint8_t expected[32] = {
        0x2B,0xDA,0x92,0x45,0x0E,0x8B,0x14,0x7F,
        0x8A,0x7C,0xB6,0x29,0xE7,0x84,0xA0,0x58,
        0xEF,0xCA,0x7C,0xF7,0xD8,0x21,0x8E,0x02,
        0xD3,0x45,0xDF,0xAA,0x65,0x24,0x4A,0x1F
    };
    uint8_t* M = make_pattern(1);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=ptn(1), C='', L=32)", out, expected, 32,
                      M, 1, NULL, 0, out, 32);
    free(M);
}

/* KangarooTwelve(M=ptn(17), C=``, 32) */
static void test_ptn17(void)
{
    const uint8_t expected[32] = {
        0x6B,0xF7,0x5F,0xA2,0x23,0x91,0x98,0xDB,
        0x47,0x72,0xE3,0x64,0x78,0xF8,0xE1,0x9B,
        0x0F,0x37,0x12,0x05,0xF6,0xA9,0xA9,0x3A,
        0x27,0x3F,0x51,0xDF,0x37,0x12,0x28,0x88
    };
    uint8_t* M = make_pattern(17);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=ptn(17), C='', L=32)", out, expected, 32,
                      M, 17, NULL, 0, out, 32);
    free(M);
}

/* KangarooTwelve(M=ptn(17^2), C=``, 32) */
static void test_ptn17sq(void)
{
    const uint8_t expected[32] = {
        0x0C,0x31,0x5E,0xBC,0xDE,0xDB,0xF6,0x14,
        0x26,0xDE,0x7D,0xCF,0x8F,0xB7,0x25,0xD1,
        0xE7,0x46,0x75,0xD7,0xF5,0x32,0x7A,0x50,
        0x67,0xF3,0x67,0xB1,0x08,0xEC,0xB6,0x7C
    };
    size_t len = 17*17;
    uint8_t* M = make_pattern(len);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=ptn(17^2), C='', L=32)", out, expected, 32,
                      M, len, NULL, 0, out, 32);
    free(M);
}

/* KangarooTwelve(M=ptn(17^3), C=``, 32) - crosses the 8192 chunk boundary */
static void test_ptn17cube(void)
{
    const uint8_t expected[32] = {
        0xCB,0x55,0x2E,0x2E,0xC7,0x7D,0x99,0x10,
        0x70,0x1D,0x57,0x8B,0x45,0x7D,0xDF,0x77,
        0x2C,0x12,0xE3,0x22,0xE4,0xEE,0x7F,0xE4,
        0x17,0xF9,0x2C,0x75,0x8F,0x0D,0x59,0xD0
    };
    size_t len = 17*17*17; /* 4913 bytes - single block */
    uint8_t* M = make_pattern(len);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=ptn(17^3), C='', L=32)", out, expected, 32,
                      M, len, NULL, 0, out, 32);
    free(M);
}

/* KangarooTwelve(M=ptn(17^4), C=``, 32) - 83521 bytes, multi-block */
static void test_ptn17_4(void)
{
    const uint8_t expected[32] = {
        0x87,0x01,0x04,0x5E,0x22,0x20,0x53,0x45,
        0xFF,0x4D,0xDA,0x05,0x55,0x5C,0xBB,0x5C,
        0x3A,0xF1,0xA7,0x71,0xC2,0xB8,0x9B,0xAE,
        0xF3,0x7D,0xB4,0x3D,0x99,0x98,0xB9,0xFE
    };
    size_t len = 17UL*17*17*17; /* 83521 bytes */
    uint8_t* M = make_pattern(len);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=ptn(17^4), C='', L=32)", out, expected, 32,
                      M, len, NULL, 0, out, 32);
    free(M);
}

/* KangarooTwelve(M=ptn(17^5), C=``, 32) */
static void test_ptn17_5(void)
{
    const uint8_t expected[32] = {
        0x84,0x4D,0x61,0x09,0x33,0xB1,0xB9,0x96,
        0x3C,0xBD,0xEB,0x5A,0xE3,0xB6,0xB0,0x5C,
        0xC7,0xCB,0xD6,0x7C,0xEE,0xDF,0x88,0x3E,
        0xB6,0x78,0xA0,0xA8,0xE0,0x37,0x16,0x82
    };
    size_t len = 17UL*17*17*17*17; /* 1419857 bytes */
    uint8_t* M = make_pattern(len);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=ptn(17^5), C='', L=32)", out, expected, 32,
                      M, len, NULL, 0, out, 32);
    free(M);
}

/* KangarooTwelve(M=ptn(17^6), C=``, 32) */
static void test_ptn17_6(void)
{
    const uint8_t expected[32] = {
        0x3C,0x39,0x07,0x82,0xA8,0xA4,0xE8,0x9F,
        0xA6,0x36,0x7F,0x72,0xFE,0xAA,0xF1,0x32,
        0x55,0xC8,0xD9,0x58,0x78,0x48,0x1D,0x3C,
        0xD8,0xCE,0x85,0xF5,0x8E,0x88,0x0A,0xF8
    };
    size_t len = 17UL*17*17*17*17*17; /* 24137569 bytes */
    uint8_t* M = make_pattern(len);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=ptn(17^6), C='', L=32)", out, expected, 32,
                      M, len, NULL, 0, out, 32);
    free(M);
}

/* KangarooTwelve(M=``, C=ptn(1), 32) - customization string */
static void test_custom_1(void)
{
    const uint8_t expected[32] = {
        0xFA,0xB6,0x58,0xDB,0x63,0xE9,0x4A,0x24,
        0x61,0x88,0xBF,0x7A,0xF6,0x9A,0x13,0x30,
        0x45,0xF4,0x6E,0xE9,0x84,0xC5,0x6E,0x3C,
        0x33,0x28,0xCA,0xAF,0x1A,0xA1,0xA5,0x83
    };
    uint8_t* C = make_pattern(1);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M='', C=ptn(1), L=32)", out, expected, 32,
                      NULL, 0, C, 1, out, 32);
    free(C);
}

/* KangarooTwelve(M=`FF`, C=ptn(41), 32) */
static void test_custom_41(void)
{
    const uint8_t expected[32] = {
        0xD8,0x48,0xC5,0x06,0x8C,0xED,0x73,0x6F,
        0x44,0x62,0x15,0x9B,0x98,0x67,0xFD,0x4C,
        0x20,0xB8,0x08,0xAC,0xC3,0xD5,0xBC,0x48,
        0xE0,0xB0,0x6B,0xA0,0xA3,0x76,0x2E,0xC4
    };
    uint8_t M[1] = { 0xFF };
    uint8_t* C = make_pattern(41);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=FF, C=ptn(41), L=32)", out, expected, 32,
                      M, 1, C, 41, out, 32);
    free(C);
}

/* KangarooTwelve(M=`FF FF FF`, C=ptn(41^2), 32) */
static void test_custom_41sq(void)
{
    const uint8_t expected[32] = {
        0xC3,0x89,0xE5,0x00,0x9A,0xE5,0x71,0x20,
        0x85,0x4C,0x2E,0x8C,0x64,0x67,0x0A,0xC0,
        0x13,0x58,0xCF,0x4C,0x1B,0xAF,0x89,0x44,
        0x7A,0x72,0x42,0x34,0xDC,0x7C,0xED,0x74
    };
    uint8_t M[3] = { 0xFF, 0xFF, 0xFF };
    size_t C_len = 41*41;
    uint8_t* C = make_pattern(C_len);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=FF*3, C=ptn(41^2), L=32)", out, expected, 32,
                      M, 3, C, C_len, out, 32);
    free(C);
}

/* KangarooTwelve(M=`FF`*7, C=ptn(41^3), 32) */
static void test_custom_41cube(void)
{
    const uint8_t expected[32] = {
        0x75,0xD2,0xF8,0x6A,0x2E,0x64,0x45,0x66,
        0x72,0x6B,0x4F,0xBC,0xFC,0x56,0x57,0xB9,
        0xDB,0xCF,0x07,0x0C,0x7B,0x0D,0xCA,0x06,
        0x45,0x0A,0xB2,0x91,0xD7,0x44,0x3B,0xCF
    };
    uint8_t M[7] = { 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF };
    size_t C_len = 41*41*41;
    uint8_t* C = make_pattern(C_len);
    uint8_t out[32];
    RUN_K12_AND_CHECK("K12(M=FF*7, C=ptn(41^3), L=32)", out, expected, 32,
                      M, 7, C, C_len, out, 32);
    free(C);
}

/* Benchmark: KangarooTwelve(M=ptn(24KB), C=``, L=32) */
static void test_benchmark_24kb(void)
{
    const size_t len = 24 * 1024;
    uint8_t* M = make_pattern(len);
    uint8_t out[32];

    const auto t0 = std::chrono::steady_clock::now();
    kangaroo_twelve(M, len, NULL, 0, out, 32);
    const auto elapsed = std::chrono::steady_clock::now() - t0;

    printf("[BENCH] K12(M=ptn(24KB), C='', L=32)\n");
    print_duration("k12 time", elapsed);

    free(M);
}

/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */
int main(void)
{
    printf("=== KangarooTwelve test vectors (RFC 9861 / draft-07) ===\n\n");

    test_empty_32();
    test_empty_64();
    test_empty_10032_last32();
    test_ptn1();
    test_ptn17();
    test_ptn17sq();
    test_ptn17cube();
    test_ptn17_4();
    test_ptn17_5();
    test_ptn17_6();
    test_custom_1();
    test_custom_41();
    test_custom_41sq();
    test_custom_41cube();

    printf("\n=== Timing (all %d test vectors) ===\n", tests_run);
    print_duration("total ", total_k12_time);

    printf("\n=== Benchmarks ===\n\n");
    test_benchmark_24kb();

    printf("\n=== Results: %d / %d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
