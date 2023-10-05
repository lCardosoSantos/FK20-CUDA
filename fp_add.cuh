// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_ADD

/**
 * @brief PTX macro for addition of two residues modulo p. Z ← X+Y
 * Z, Y and X may be the same.
 */
#define FP_ADD(Z, X, Y) \
    /* z = x + y */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#X"0, "#Y"0;" \
    "\n\taddc.u64.cc "#Z"1, "#X"1, "#Y"1;" \
    "\n\taddc.u64.cc "#Z"2, "#X"2, "#Y"2;" \
    "\n\taddc.u64.cc "#Z"3, "#X"3, "#Y"3;" \
    "\n\taddc.u64.cc "#Z"4, "#X"4, "#Y"4;" \
    "\n\taddc.u64.cc "#Z"5, "#X"5, "#Y"5;" \
    "\n\taddc.u64    "#Z"6,  0,   0;" \
 \
    /* gt = (z>>320) > (m>>320) */ \
    /* nz = (z>>384) > 0 */ \
 \
    "\n\tsetp.gt.u64 gt, "#Z"5, 0x1a0111ea397fe69aU;" \
    "\n\tsetp.ne.u64 nz, "#Z"6, 0;" \
 \
    /* If gt then subtract m */ \
 \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0xb9feffffffffaaabU;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x1eabfffeb153ffffU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0x6730d2a0f6b0f624U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x64774b84f38512bfU;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0x4b1ba7b6434bacd7U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0x1a0111ea397fe69aU;" \
 \
    /* If nz then subtract mmu0 (= 9m) */ \
 \
    "\n@nz\tsub.u64.cc  "#Z"0, "#Z"0, 0x89f6fffffffd0003U;" \
    "\n@nz\tsubc.u64.cc "#Z"1, "#Z"1, 0x140bfff43bf3fffdU;" \
    "\n@nz\tsubc.u64.cc "#Z"2, "#Z"2, 0xa0b767a8ac38a745U;" \
    "\n@nz\tsubc.u64.cc "#Z"3, "#Z"3, 0x8831a7ac8fada8baU;" \
    "\n@nz\tsubc.u64.cc "#Z"4, "#Z"4, 0xa3f8e5685da91392U;" \
    "\n@nz\tsubc.u64.cc "#Z"5, "#Z"5, 0xea09a13c057f1b6cU;"

__forceinline__
__device__ void fp_add(
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
    uint64_t x5,
    uint64_t y0,
    uint64_t y1,
    uint64_t y2,
    uint64_t y3,
    uint64_t y4,
    uint64_t y5) {

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 z<7>, x<6>, y<6>;"
    "\n\t.reg .pred gt, nz;"

    "\n\tmov.u64 x0,  %6;"
    "\n\tmov.u64 x1,  %7;"
    "\n\tmov.u64 x2,  %8;"
    "\n\tmov.u64 x3,  %9;"
    "\n\tmov.u64 x4, %10;"
    "\n\tmov.u64 x5, %11;"

    "\n\tmov.u64 y0, %12;"
    "\n\tmov.u64 y1, %13;"
    "\n\tmov.u64 y2, %14;"
    "\n\tmov.u64 y3, %15;"
    "\n\tmov.u64 y4, %16;"
    "\n\tmov.u64 y5, %17;"

FP_ADD(z, x, y)

    "\n\tmov.u64 %0,  z0;"
    "\n\tmov.u64 %1,  z1;"
    "\n\tmov.u64 %2,  z2;"
    "\n\tmov.u64 %3,  z3;"
    "\n\tmov.u64 %4,  z4;"
    "\n\tmov.u64 %5,  z5;"

    "\n\t}"
    :
    "=l"(z0), "=l"(z1), "=l"(z2), "=l"(z3), "=l"(z4), "=l"(z5)
    :
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5),
    "l"(y0), "l"(y1), "l"(y2), "l"(y3), "l"(y4), "l"(y5)
    );
}

#endif
// vim: ts=4 et sw=4 si
