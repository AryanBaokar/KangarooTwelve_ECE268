# DELETE THIS LATER
"""
Test harness for keccak_p[1600, 12] (rounds 12-23).

Strategy:
  - Reference implementation of Keccak-f[1600] (all 24 rounds) in pure Python
  - For each test input, run rounds 0-11 to get the split-point state,
    then run rounds 12-23 to get the expected output.
  - The C++ keccak_p() is loaded via ctypes and called directly.

Usage:
  # Compile your shared library first:
  g++ -O2 -shared -fPIC -o libkeccak_p.so keccak_p.cpp

  # Run all tests:
  python3 test_keccak_p.py

  # Point to a custom .so path:
  python3 test_keccak_p.py --lib /path/to/libkeccak_p.so

  # Self-test the Python reference only (no C++ needed):
  python3 test_keccak_p.py --selftest

  # Print C++ arrays for all vectors:
  python3 test_keccak_p.py --generate
"""

import sys
import ctypes
import os
import argparse

# ---------------------------------------------------------------------------
# Python reference: Keccak-f[1600]
# ---------------------------------------------------------------------------

RHO_OFFSETS = [
    [ 0,  1, 62, 28, 27],
    [36, 44,  6, 55, 20],
    [ 3, 10, 43, 25, 39],
    [41, 45, 15, 21,  8],
    [18,  2, 61, 56, 14],
]

ROUND_CONSTANTS_ALL = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

M64 = 0xFFFFFFFFFFFFFFFF


def rot64(x, n):
    return ((x << n) | (x >> (64 - n))) & M64


def keccak_round(A, rc):
    # Theta
    C = [A[0][x] ^ A[1][x] ^ A[2][x] ^ A[3][x] ^ A[4][x] for x in range(5)]
    D = [C[(x - 1) % 5] ^ rot64(C[(x + 1) % 5], 1) for x in range(5)]
    A = [[A[y][x] ^ D[x] for x in range(5)] for y in range(5)]
    # Rho + Pi
    B = [[0] * 5 for _ in range(5)]
    for y in range(5):
        for x in range(5):
            x_old = (x + 3 * y) % 5
            y_old = x
            B[y][x] = rot64(A[y_old][x_old], RHO_OFFSETS[y_old][x_old])
    # Chi
    A = [
        [(B[y][x] ^ ((~B[y][(x + 1) % 5]) & B[y][(x + 2) % 5])) & M64
         for x in range(5)]
        for y in range(5)
    ]
    # Iota
    A[0][0] ^= rc
    return A


def keccak_f1600(state_flat, start_round=0, end_round=24):
    A = [[state_flat[x + 5 * y] for x in range(5)] for y in range(5)]
    for rnd in range(start_round, end_round):
        A = keccak_round(A, ROUND_CONSTANTS_ALL[rnd])
    return [A[y][x] for y in range(5) for x in range(5)]


def keccak_p_reference(state_flat):
    """Python reference: keccak_p[1600,12] = rounds 12..23."""
    return keccak_f1600(state_flat, start_round=12, end_round=24)


# ---------------------------------------------------------------------------
# ctypes bridge to C++ keccak_p
# ---------------------------------------------------------------------------

def load_cpp_lib(so_path):
    """Load libkeccak_p.so and return a callable that wraps keccak_p()."""
    if not os.path.exists(so_path):
        print(f"Error: shared library not found: {so_path}")
        print("Compile it with:")
        print("  g++ -O2 -shared -fPIC -o libkeccak_p.so keccak_p.cpp")
        sys.exit(1)

    lib = ctypes.CDLL(so_path)
    # keccak_p is a C++ function so its symbol is mangled.
    # We expose it via an extern "C" wrapper compiled into the same .so:
    #   extern "C" void keccak_p_c(uint64_t state[25]) { keccak_p(state); }
    fn = lib.keccak_p_c
    fn.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    fn.restype  = None

    def keccak_p_cpp(state_flat):
        """Call C++ keccak_p on a list of 25 ints, return list of 25 ints."""
        arr = (ctypes.c_uint64 * 25)(*state_flat)
        fn(arr)
        return list(arr)

    return keccak_p_cpp


# ---------------------------------------------------------------------------
# Test vectors
# ---------------------------------------------------------------------------

def make_test_vectors():
    vectors = []

    def add(name, seed_flat):
        inp = keccak_f1600(seed_flat, start_round=0, end_round=12)
        out = keccak_p_reference(inp)
        vectors.append((name, inp, out))

    add("all_zeros",           [0] * 25)
    add("all_ones",            [M64] * 25)

    s = [0] * 25; s[0] = 1
    add("single_bit_lane0",    s)

    s = [0] * 25; s[24] = 1
    add("single_bit_lane24",   s)

    add("alternating_bits",
        [0x5555555555555555 if i % 2 == 0 else 0xAAAAAAAAAAAAAAAA
         for i in range(25)])

    add("incrementing",        list(range(1, 26)))

    add("byte_pattern",
        [i * 0x0101010101010101 & M64 for i in range(25)])

    add("high_bits_only",      [0x8000000000000000] * 25)

    s = [0] * 25
    for i in range(5):
        s[i + 5 * i] = M64
    add("diagonal_lanes",      s)

    s = [0] * 25
    s[0]  = 0x0600000000000000
    s[16] = 0x8000000000000000
    add("sha3_padding_pattern", s)

    return vectors


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def cmd_test(keccak_p_cpp, vectors):
    """Run all vectors through the C++ implementation and compare."""
    print(f"Testing C++ keccak_p against {len(vectors)} vectors...\n")
    all_ok = True

    for name, inp, expected in vectors:
        got = keccak_p_cpp(inp)
        ok  = (got == expected)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_ok = False
            for i in range(25):
                if got[i] != expected[i]:
                    print(f"         Lane[{i}] (y={i//5} x={i%5}): "
                          f"got      0x{got[i]:016X}\n"
                          f"         {'':38}expected 0x{expected[i]:016X}")

    print()
    if all_ok:
        print(f"All {len(vectors)} vectors passed!")
    else:
        print("Some vectors FAILED — see lane differences above.")
    return all_ok


def cmd_selftest(vectors):
    """Verify the Python reference is internally consistent (no C++ needed)."""
    print("Running Python reference self-tests...\n")
    all_ok = True
    for name, inp, expected in vectors:
        got = keccak_p_reference(inp)
        ok  = (got == expected)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            all_ok = False
            for i in range(25):
                if got[i] != expected[i]:
                    print(f"         Lane[{i}]: got 0x{got[i]:016X} "
                          f"expected 0x{expected[i]:016X}")
    print()
    if all_ok:
        print("Python reference is consistent.")
    else:
        print("SELF-TEST FAILURES — bug in the Python reference!")
    return all_ok


def cmd_generate(vectors):
    """Print C++ arrays for all vectors."""
    print("// Auto-generated by test_keccak_p.py --generate\n")
    print(f"static const int NUM_VECTORS = {len(vectors)};\n")
    print("static const char* VECTOR_NAMES[] = {")
    for name, _, _ in vectors:
        print(f'    "{name}",')
    print("};\n")
    for idx, (name, inp, out) in enumerate(vectors):
        print(f"// Vector {idx}: {name}")
        for label, arr in [("INPUT", inp), ("EXPECTED", out)]:
            print(f"static const uint64_t {label}_{idx}[25] = {{")
            for i, v in enumerate(arr):
                comma = "," if i < 24 else ""
                print(f"    0x{v:016X}ULL{comma}  // [{i}]")
            print("};\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test harness for keccak_p[1600, 12]"
    )
    parser.add_argument(
        "--lib", default="./libkeccak_p.so",
        help="Path to compiled shared library (default: ./libkeccak_p.so)"
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Self-test the Python reference only (no C++ needed)"
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Print C++ test arrays for all vectors"
    )
    args = parser.parse_args()

    vectors = make_test_vectors()

    if args.selftest:
        ok = cmd_selftest(vectors)
        sys.exit(0 if ok else 1)

    if args.generate:
        cmd_generate(vectors)
        sys.exit(0)

    # Default: load C++ lib and run all tests
    keccak_p_cpp = load_cpp_lib(args.lib)
    ok = cmd_test(keccak_p_cpp, vectors)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()