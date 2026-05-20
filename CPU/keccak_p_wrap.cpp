// DELETE THIS LATER
//2. Compile both files into a shared library
//g++ -O2 -shared -fPIC -o libkeccak_p.so keccak_p.cpp keccak_p_wrap.cpp
//
//3. Run the tests
//python3 test_keccak_p.py

#include "keccak_p.h"
extern "C" {
    void keccak_p_c(uint64_t state[25]) { keccak_p(state); }
}
