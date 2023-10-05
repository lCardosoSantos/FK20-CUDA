// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_NEG

/**
 * @brief PTX macro for compuing an additive inverse of x modulo p. Z ← -X
 * There are no restrictions on X and Z being the same.
 *
 * Z must have an extra word to handle underflow.
 */
#define FP_NEG(Z, X) \
    /* z = pmmu0 - x */ \
\
    "\n\tsub.u64.cc  "#Z"0, 0x89F6FFFFFFFD0003U, "#X"0;" \
    "\n\tsubc.u64.cc "#Z"1, 0x140BFFF43BF3FFFDU, "#X"1;" \
    "\n\tsubc.u64.cc "#Z"2, 0xA0B767A8AC38A745U, "#X"2;" \
    "\n\tsubc.u64.cc "#Z"3, 0x8831A7AC8FADA8BAU, "#X"3;" \
    "\n\tsubc.u64.cc "#Z"4, 0xA3F8E5685DA91392U, "#X"4;" \
    "\n\tsubc.u64.cc "#Z"5, 0xEA09A13C057F1B6CU, "#X"5;" \
    "\n\tsubc.u64    "#Z"6,  0, 0;" \
    "\n\tsetp.ne.u64 nz, "#Z"6, 0;" \
\
    /* if nz (borrow) then z += p */ \
\
    "\n@nz\tadd.u64.cc  "#Z"0, "#Z"0, 0xB9FEFFFFFFFFAAABU;" \
    "\n@nz\taddc.u64.cc "#Z"1, "#Z"1, 0x1EABFFFEB153FFFFU;" \
    "\n@nz\taddc.u64.cc "#Z"2, "#Z"2, 0x6730D2A0F6B0F624U;" \
    "\n@nz\taddc.u64.cc "#Z"3, "#Z"3, 0x64774B84F38512BFU;" \
    "\n@nz\taddc.u64.cc "#Z"4, "#Z"4, 0x4B1BA7B6434BACD7U;" \
    "\n@nz\taddc.u64    "#Z"5, "#Z"5, 0x1A0111EA397FE69AU;"

__forceinline__
__device__ void fp_neg(
    uint64_t &z0,
    uint64_t &z1,
    uint64_t &z2,
    uint64_t &z3,
    uint64_t &z4,
    uint64_t &z5,
    uint64_t x0,
    uint64_t x1,
    uint64_t x2,
    uint64_t x3,
    uint64_t x4,
    uint64_t x5) {

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 z<7>, x<6>;"
    "\n\t.reg .pred nz;"

    "\n\tmov.u64 x0,  %6;"
    "\n\tmov.u64 x1,  %7;"
    "\n\tmov.u64 x2,  %8;"
    "\n\tmov.u64 x3,  %9;"
    "\n\tmov.u64 x4, %10;"
    "\n\tmov.u64 x5, %11;"

FP_NEG(z, x)

    "\n\tmov.u64 %0, z0;"
    "\n\tmov.u64 %1, z1;"
    "\n\tmov.u64 %2, z2;"
    "\n\tmov.u64 %3, z3;"
    "\n\tmov.u64 %4, z4;"
    "\n\tmov.u64 %5, z5;"

    "\n\t}"
    :
    "=l"(z0), "=l"(z1), "=l"(z2), "=l"(z3), "=l"(z4), "=l"(z5)
    :
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5)
    );
}

#endif
// vim: ts=4 et sw=4 si
