#ifndef FP_MUL

/**
 * @brief PTX macro for multiplication of two residues mod p. Z ← X*Y
 * Z may NOT be the same as X or Y.
 * X and Y may be the same.
 */
#define FP_MUL(Z, X, Y) \
    "\n\tmul.lo.u64     "#Z"1, "#X"0, "#Y"1       ; mul.hi.u64     "#Z"2, "#X"0, "#Y"1       ;" \
    "\n\tmul.lo.u64     "#Z"3, "#X"0, "#Y"3       ; mul.hi.u64     "#Z"4, "#X"0, "#Y"3       ;" \
    "\n\tmul.lo.u64     "#Z"5, "#X"0, "#Y"5       ; mul.hi.u64     "#Z"6, "#X"0, "#Y"5       ;" \
\
    "\n\tmul.lo.u64     "#Z"0, "#X"0, "#Y"0       ; mad.hi.u64.cc  "#Z"1, "#X"0, "#Y"0, "#Z"1;" \
    "\n\tmadc.lo.u64.cc "#Z"2, "#X"0, "#Y"2, "#Z"2; madc.hi.u64.cc "#Z"3, "#X"0, "#Y"2, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"0, "#Y"4, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"0, "#Y"4, "#Z"5;" \
    "\n\taddc.u64       "#Z"6,     0, "#Z"6;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"2, "#X"1, "#Y"1, "#Z"2; madc.hi.u64.cc "#Z"3, "#X"1, "#Y"1, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"1, "#Y"3, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"1, "#Y"3, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"1, "#Y"5, "#Z"6; madc.hi.u64    "#Z"7, "#X"1, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"1, "#X"1, "#Y"0, "#Z"1; madc.hi.u64.cc "#Z"2, "#X"1, "#Y"0, "#Z"2;" \
    "\n\tmadc.lo.u64.cc "#Z"3, "#X"1, "#Y"2, "#Z"3; madc.hi.u64.cc "#Z"4, "#X"1, "#Y"2, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"1, "#Y"4, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"1, "#Y"4, "#Z"6;" \
    "\n\taddc.u64       "#Z"7,     0, "#Z"7;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"3, "#X"2, "#Y"1, "#Z"3; madc.hi.u64.cc "#Z"4, "#X"2, "#Y"1, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"2, "#Y"3, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"2, "#Y"3, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"2, "#Y"5, "#Z"7; madc.hi.u64    "#Z"8, "#X"2, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"2, "#X"2, "#Y"0, "#Z"2; madc.hi.u64.cc "#Z"3, "#X"2, "#Y"0, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"2, "#Y"2, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"2, "#Y"2, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"2, "#Y"4, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"2, "#Y"4, "#Z"7;" \
    "\n\taddc.u64       "#Z"8,     0, "#Z"8;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"4, "#X"3, "#Y"1, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"3, "#Y"1, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"3, "#Y"3, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"3, "#Y"3, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"3, "#Y"5, "#Z"8; madc.hi.u64    "#Z"9, "#X"3, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"3, "#X"3, "#Y"0, "#Z"3; madc.hi.u64.cc "#Z"4, "#X"3, "#Y"0, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"3, "#Y"2, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"3, "#Y"2, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"3, "#Y"4, "#Z"7; madc.hi.u64.cc "#Z"8, "#X"3, "#Y"4, "#Z"8;" \
    "\n\taddc.u64       "#Z"9,     0, "#Z"9;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"5, "#X"4, "#Y"1, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"4, "#Y"1, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"4, "#Y"3, "#Z"7; madc.hi.u64.cc "#Z"8, "#X"4, "#Y"3, "#Z"8;" \
    "\n\tmadc.lo.u64.cc "#Z"9, "#X"4, "#Y"5, "#Z"9; madc.hi.u64    "#Z"a, "#X"4, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"4, "#X"4, "#Y"0, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"4, "#Y"0, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"4, "#Y"2, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"4, "#Y"2, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"4, "#Y"4, "#Z"8; madc.hi.u64.cc "#Z"9, "#X"4, "#Y"4, "#Z"9;" \
    "\n\taddc.u64       "#Z"a,     0, "#Z"a;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"6, "#X"5, "#Y"1, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"5, "#Y"1, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"5, "#Y"3, "#Z"8; madc.hi.u64.cc "#Z"9, "#X"5, "#Y"3, "#Z"9;" \
    "\n\tmadc.lo.u64.cc "#Z"a, "#X"5, "#Y"5, "#Z"a; madc.hi.u64    "#Z"b, "#X"5, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"5, "#X"5, "#Y"0, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"5, "#Y"0, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"5, "#Y"2, "#Z"7; madc.hi.u64.cc "#Z"8, "#X"5, "#Y"2, "#Z"8;" \
    "\n\tmadc.lo.u64.cc "#Z"9, "#X"5, "#Y"4, "#Z"9; madc.hi.u64.cc "#Z"a, "#X"5, "#Y"4, "#Z"a;" \
    "\n\taddc.u64       "#Z"b,     0, "#Z"b;\n"

__forceinline__
__device__ void fp_mul(
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
    uint64_t x5,
    uint64_t y0,
    uint64_t y1,
    uint64_t y2,
    uint64_t y3,
    uint64_t y4,
    uint64_t y5) {

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 z<10>, za, zb, x<6>, y<6>;"

    "\n\tmov.u64 x0, %12;"
    "\n\tmov.u64 x1, %13;"
    "\n\tmov.u64 x2, %14;"
    "\n\tmov.u64 x3, %15;"
    "\n\tmov.u64 x4, %16;"
    "\n\tmov.u64 x5, %17;"

    "\n\tmov.u64 y0, %18;"
    "\n\tmov.u64 y1, %19;"
    "\n\tmov.u64 y2, %20;"
    "\n\tmov.u64 y3, %21;"
    "\n\tmov.u64 y4, %22;"
    "\n\tmov.u64 y5, %23;"

FP_MUL(z, x, y)

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
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5),
    "l"(y0), "l"(y1), "l"(y2), "l"(y3), "l"(y4), "l"(y5)
    );
}

#endif
// vim: ts=4 et sw=4 si
