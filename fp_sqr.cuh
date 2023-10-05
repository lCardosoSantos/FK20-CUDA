// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_SQR

/**
 * @brief PTX macro for computing the square of the residue x modulo p. Z ← X*X
 * Z and X may NOT be the same.
 *
 */
#define FP_SQR(Z, X) \
\
    "\n\tmul.lo.u64     "#Z"5, "#X"0, "#X"5       ; mul.hi.u64     "#Z"6, "#X"0, "#X"5       ;" \
\
    "\n\tmul.lo.u64     "#Z"4, "#X"0, "#X"4       ; mad.hi.u64.cc  "#Z"5, "#X"0, "#X"4, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"1, "#X"5, "#Z"6; madc.hi.u64    "#Z"7, "#X"1, "#X"5,     0;" \
\
    "\n\tmul.lo.u64     "#Z"3, "#X"0, "#X"3       ; mad.hi.u64.cc  "#Z"4, "#X"0, "#X"3, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"1, "#X"4, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"1, "#X"4, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"2, "#X"5, "#Z"7; madc.hi.u64    "#Z"8, "#X"2, "#X"5,     0;" \
\
    "\n\tmul.lo.u64     "#Z"2, "#X"0, "#X"2       ; mad.hi.u64.cc  "#Z"3, "#X"0, "#X"2, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"1, "#X"3, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"1, "#X"3, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"2, "#X"4, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"2, "#X"4, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"3, "#X"5, "#Z"8; madc.hi.u64    "#Z"9, "#X"3, "#X"5,     0;" \
\
    "\n\tmul.lo.u64     "#Z"1, "#X"0, "#X"1       ; mad.hi.u64.cc  "#Z"2, "#X"0, "#X"1, "#Z"2;" \
    "\n\tmadc.lo.u64.cc "#Z"3, "#X"1, "#X"2, "#Z"3; madc.hi.u64.cc "#Z"4, "#X"1, "#X"2, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"2, "#X"3, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"2, "#X"3, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"3, "#X"4, "#Z"7; madc.hi.u64.cc "#Z"8, "#X"3, "#X"4, "#Z"8;" \
    "\n\tmadc.lo.u64.cc "#Z"9, "#X"4, "#X"5, "#Z"9; madc.hi.u64    "#Z"a, "#X"4, "#X"5,     0;" \
\
    "\n\tadd.u64.cc  "#Z"1, "#Z"1, "#Z"1;" \
    "\n\taddc.u64.cc "#Z"2, "#Z"2, "#Z"2;" \
    "\n\taddc.u64.cc "#Z"3, "#Z"3, "#Z"3;" \
    "\n\taddc.u64.cc "#Z"4, "#Z"4, "#Z"4;" \
    "\n\taddc.u64.cc "#Z"5, "#Z"5, "#Z"5;" \
    "\n\taddc.u64.cc "#Z"6, "#Z"6, "#Z"6;" \
    "\n\taddc.u64.cc "#Z"7, "#Z"7, "#Z"7;" \
    "\n\taddc.u64.cc "#Z"8, "#Z"8, "#Z"8;" \
    "\n\taddc.u64.cc "#Z"9, "#Z"9, "#Z"9;" \
    "\n\taddc.u64.cc "#Z"a, "#Z"a, "#Z"a;" \
    "\n\taddc.u64    "#Z"b,     0,     0;" \
\
    "\n\tmul.lo.u64     "#Z"0, "#X"0, "#X"0       ; mad.hi.u64.cc  "#Z"1, "#X"0, "#X"0, "#Z"1;" \
    "\n\tmadc.lo.u64.cc "#Z"2, "#X"1, "#X"1, "#Z"2; madc.hi.u64.cc "#Z"3, "#X"1, "#X"1, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"2, "#X"2, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"2, "#X"2, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"3, "#X"3, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"3, "#X"3, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"4, "#X"4, "#Z"8; madc.hi.u64.cc "#Z"9, "#X"4, "#X"4, "#Z"9;" \
    "\n\tmadc.lo.u64.cc "#Z"a, "#X"5, "#X"5, "#Z"a; madc.hi.u64    "#Z"b, "#X"5, "#X"5, "#Z"b;"

__forceinline__
__device__ void fp_sqr(
    uint64_t &z0,
    uint64_t &z1,
    uint64_t &z2,
    uint64_t &z3,
    uint64_t &z4,
    uint64_t &z5,
    uint64_t &z6,
    uint64_t &z7,
    uint64_t &z8,
    uint64_t &z9,
    uint64_t &za,
    uint64_t &zb,
    uint64_t x0,
    uint64_t x1,
    uint64_t x2,
    uint64_t x3,
    uint64_t x4,
    uint64_t x5) {

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 z<10>, za, zb, x<6>;"

    "\n\tmov.u64 x0, %12;"
    "\n\tmov.u64 x1, %13;"
    "\n\tmov.u64 x2, %14;"
    "\n\tmov.u64 x3, %15;"
    "\n\tmov.u64 x4, %16;"
    "\n\tmov.u64 x5, %17;"

FP_SQR(z, x)

    "\n\tmov.u64  %0, z0;"
    "\n\tmov.u64  %1, z1;"
    "\n\tmov.u64  %2, z2;"
    "\n\tmov.u64  %3, z3;"
    "\n\tmov.u64  %4, z4;"
    "\n\tmov.u64  %5, z5;"
    "\n\tmov.u64  %6, z6;"
    "\n\tmov.u64  %7, z7;"
    "\n\tmov.u64  %8, z8;"
    "\n\tmov.u64  %9, z9;"
    "\n\tmov.u64 %10, za;"
    "\n\tmov.u64 %11, zb;"

    "\n\t}"
    :
    "=l"(z0), "=l"(z1), "=l"(z2), "=l"(z3), "=l"(z4), "=l"(z5),
    "=l"(z6), "=l"(z7), "=l"(z8), "=l"(z9), "=l"(za), "=l"(zb)
    :
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5)
    );
}

#endif
// vim: ts=4 et sw=4 si
