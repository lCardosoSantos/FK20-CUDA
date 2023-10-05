// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_REDUCE12

/**
 * @brief Wide reduction over 12 words. Z ← Z%p
 * Reads Z0..Zb.
 * Writes Z0..Z5.
 * Modifies Q0..Q7 and S0..S6.
 */
#define FP_REDUCE12(Z, Q, S) \
\
    /* q2 = q1 * mu; q3 = q2 / 2^448 */ \
\
    /* mu0 */ \
\
    "\n\tmul.hi.u64     "#Q"0, 0x13E207F56591BA2EU, "#Z"a;" \
\
    "\n\tmad.lo.u64.cc  "#Q"0, 0x13E207F56591BA2EU, "#Z"b, "#Q"0;" \
    "\n\tmadc.hi.u64    "#Q"1, 0x13E207F56591BA2EU, "#Z"b,     0;" \
\
    /* mu1 */ \
\
    "\n\tmad.hi.u64.cc  "#Q"0, 0x997167A058F1C07BU, "#Z"9, "#Q"0;" \
    "\n\tmadc.lo.u64.cc "#Q"1, 0x997167A058F1C07BU, "#Z"b, "#Q"1;" \
    "\n\tmadc.hi.u64    "#Q"2, 0x997167A058F1C07BU, "#Z"b,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Q"0, 0x997167A058F1C07BU, "#Z"a, "#Q"0;" \
    "\n\tmadc.hi.u64.cc "#Q"1, 0x997167A058F1C07BU, "#Z"a, "#Q"1;" \
    "\n\taddc.u64       "#Q"2, "#Q"2, 0;" \
\
    /* mu2 */ \
\
    "\n\tmad.lo.u64.cc  "#Q"0, 0xDF4771E0286779D3U, "#Z"9, "#Q"0;" \
    "\n\tmadc.hi.u64.cc "#Q"1, 0xDF4771E0286779D3U, "#Z"9, "#Q"1;" \
    "\n\tmadc.lo.u64.cc "#Q"2, 0xDF4771E0286779D3U, "#Z"b, "#Q"2;" \
    "\n\tmadc.hi.u64    "#Q"3, 0xDF4771E0286779D3U, "#Z"b,     0;" \
\
    "\n\tmad.hi.u64.cc  "#Q"0, 0xDF4771E0286779D3U, "#Z"8, "#Q"0;" \
    "\n\tmadc.lo.u64.cc "#Q"1, 0xDF4771E0286779D3U, "#Z"a, "#Q"1;" \
    "\n\tmadc.hi.u64.cc "#Q"2, 0xDF4771E0286779D3U, "#Z"a, "#Q"2;" \
    "\n\taddc.u64       "#Q"3, "#Q"3, 0;" \
\
    /* mu3 */ \
\
    "\n\tmad.hi.u64.cc  "#Q"0, 0x1B82741FF6A0A94BU, "#Z"7, "#Q"0;" \
    "\n\tmadc.lo.u64.cc "#Q"1, 0x1B82741FF6A0A94BU, "#Z"9, "#Q"1;" \
    "\n\tmadc.hi.u64.cc "#Q"2, 0x1B82741FF6A0A94BU, "#Z"9, "#Q"2;" \
    "\n\tmadc.lo.u64.cc "#Q"3, 0x1B82741FF6A0A94BU, "#Z"b, "#Q"3;" \
    "\n\tmadc.hi.u64    "#Q"4, 0x1B82741FF6A0A94BU, "#Z"b,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Q"0, 0x1B82741FF6A0A94BU, "#Z"8, "#Q"0;" \
    "\n\tmadc.hi.u64.cc "#Q"1, 0x1B82741FF6A0A94BU, "#Z"8, "#Q"1;" \
    "\n\tmadc.lo.u64.cc "#Q"2, 0x1B82741FF6A0A94BU, "#Z"a, "#Q"2;" \
    "\n\tmadc.hi.u64.cc "#Q"3, 0x1B82741FF6A0A94BU, "#Z"a, "#Q"3;" \
    "\n\taddc.u64       "#Q"4, "#Q"4, 0;" \
\
    /* mu4 */ \
\
    "\n\tmad.lo.u64.cc  "#Q"0, 0x28101B0CC7A6BA29U, "#Z"7, "#Q"0;" \
    "\n\tmadc.hi.u64.cc "#Q"1, 0x28101B0CC7A6BA29U, "#Z"7, "#Q"1;" \
    "\n\tmadc.lo.u64.cc "#Q"2, 0x28101B0CC7A6BA29U, "#Z"9, "#Q"2;" \
    "\n\tmadc.hi.u64.cc "#Q"3, 0x28101B0CC7A6BA29U, "#Z"9, "#Q"3;" \
    "\n\tmadc.lo.u64.cc "#Q"4, 0x28101B0CC7A6BA29U, "#Z"b, "#Q"4;" \
    "\n\tmadc.hi.u64    "#Q"5, 0x28101B0CC7A6BA29U, "#Z"b,     0;" \
\
    "\n\tmad.hi.u64.cc  "#Q"0, 0x28101B0CC7A6BA29U, "#Z"6, "#Q"0;" \
    "\n\tmadc.lo.u64.cc "#Q"1, 0x28101B0CC7A6BA29U, "#Z"8, "#Q"1;" \
    "\n\tmadc.hi.u64.cc "#Q"2, 0x28101B0CC7A6BA29U, "#Z"8, "#Q"2;" \
    "\n\tmadc.lo.u64.cc "#Q"3, 0x28101B0CC7A6BA29U, "#Z"a, "#Q"3;" \
    "\n\tmadc.hi.u64.cc "#Q"4, 0x28101B0CC7A6BA29U, "#Z"a, "#Q"4;" \
    "\n\taddc.u64       "#Q"5, "#Q"5, 0;" \
\
    /* mu5 */ \
\
    "\n\tmad.hi.u64.cc  "#Q"0, 0xD835D2F3CC9E45CEU, "#Z"5, "#Q"0;" \
    "\n\tmadc.lo.u64.cc "#Q"1, 0xD835D2F3CC9E45CEU, "#Z"7, "#Q"1;" \
    "\n\tmadc.hi.u64.cc "#Q"2, 0xD835D2F3CC9E45CEU, "#Z"7, "#Q"2;" \
    "\n\tmadc.lo.u64.cc "#Q"3, 0xD835D2F3CC9E45CEU, "#Z"9, "#Q"3;" \
    "\n\tmadc.hi.u64.cc "#Q"4, 0xD835D2F3CC9E45CEU, "#Z"9, "#Q"4;" \
    "\n\tmadc.lo.u64.cc "#Q"5, 0xD835D2F3CC9E45CEU, "#Z"b, "#Q"5;" \
    "\n\tmadc.hi.u64    "#Q"6, 0xD835D2F3CC9E45CEU, "#Z"b,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Q"0, 0xD835D2F3CC9E45CEU, "#Z"6, "#Q"0;" \
    "\n\tmadc.hi.u64.cc "#Q"1, 0xD835D2F3CC9E45CEU, "#Z"6, "#Q"1;" \
    "\n\tmadc.lo.u64.cc "#Q"2, 0xD835D2F3CC9E45CEU, "#Z"8, "#Q"2;" \
    "\n\tmadc.hi.u64.cc "#Q"3, 0xD835D2F3CC9E45CEU, "#Z"8, "#Q"3;" \
    "\n\tmadc.lo.u64.cc "#Q"4, 0xD835D2F3CC9E45CEU, "#Z"a, "#Q"4;" \
    "\n\tmadc.hi.u64.cc "#Q"5, 0xD835D2F3CC9E45CEU, "#Z"a, "#Q"5;" \
    "\n\taddc.u64       "#Q"6, "#Q"6, 0;" \
\
    /* mu6 */ \
\
    "\n\tmad.lo.u64.cc  "#Q"0, 0x0000000000000009U, "#Z"5, "#Q"0;" \
    "\n\tmadc.hi.u64.cc "#Q"1, 0x0000000000000009U, "#Z"5, "#Q"1;" \
    "\n\tmadc.lo.u64.cc "#Q"2, 0x0000000000000009U, "#Z"7, "#Q"2;" \
    "\n\tmadc.hi.u64.cc "#Q"3, 0x0000000000000009U, "#Z"7, "#Q"3;" \
    "\n\tmadc.lo.u64.cc "#Q"4, 0x0000000000000009U, "#Z"9, "#Q"4;" \
    "\n\tmadc.hi.u64.cc "#Q"5, 0x0000000000000009U, "#Z"9, "#Q"5;" \
    "\n\tmadc.lo.u64.cc "#Q"6, 0x0000000000000009U, "#Z"b, "#Q"6;" \
    "\n\tmadc.hi.u64    "#Q"7, 0x0000000000000009U, "#Z"b,     0;" \
\
    "\n\tmad.hi.u64.cc  "#Q"0, 0x0000000000000009U, "#Z"4, "#Q"0;" \
    "\n\tmadc.lo.u64.cc "#Q"1, 0x0000000000000009U, "#Z"6, "#Q"1;" \
    "\n\tmadc.hi.u64.cc "#Q"2, 0x0000000000000009U, "#Z"6, "#Q"2;" \
    "\n\tmadc.lo.u64.cc "#Q"3, 0x0000000000000009U, "#Z"8, "#Q"3;" \
    "\n\tmadc.hi.u64.cc "#Q"4, 0x0000000000000009U, "#Z"8, "#Q"4;" \
    "\n\tmadc.lo.u64.cc "#Q"5, 0x0000000000000009U, "#Z"a, "#Q"5;" \
    "\n\tmadc.hi.u64.cc "#Q"6, 0x0000000000000009U, "#Z"a, "#Q"6;" \
    "\n\taddc.u64       "#Q"7, "#Q"7, 0;" \
\
    /* r2 = q3 * m mod 2^448 */ \
    /*  u contains z^2 */ \
    /*  q contains q3 */ \
    /*  produces r2 in r */ \
\
    /* m5 */ \
\
    "\n\tmul.lo.u64     "#S"5, 0x1A0111EA397FE69AU, "#Q"1       ;" \
    "\n\tmul.hi.u64     "#S"6, 0x1A0111EA397FE69AU, "#Q"1       ;" \
    "\n\tmad.lo.u64     "#S"6, 0x1A0111EA397FE69AU, "#Q"2, "#S"6;" \
\
    /* m4 */ \
\
    "\n\tmul.lo.u64     "#S"4, 0x4B1BA7B6434BACD7U, "#Q"1       ;" \
    "\n\tmad.hi.u64.cc  "#S"5, 0x4B1BA7B6434BACD7U, "#Q"1, "#S"5;" \
    "\n\tmadc.lo.u64    "#S"6, 0x4B1BA7B6434BACD7U, "#Q"3, "#S"6;" \
\
    "\n\tmad.lo.u64.cc  "#S"5, 0x4B1BA7B6434BACD7U, "#Q"2, "#S"5;" \
    "\n\tmadc.hi.u64    "#S"6, 0x4B1BA7B6434BACD7U, "#Q"2, "#S"6;" \
\
    /* m3 */ \
\
    "\n\tmul.lo.u64     "#S"3, 0x64774B84F38512BFU, "#Q"1       ;" \
    "\n\tmad.hi.u64.cc  "#S"4, 0x64774B84F38512BFU, "#Q"1, "#S"4;" \
    "\n\tmadc.lo.u64.cc "#S"5, 0x64774B84F38512BFU, "#Q"3, "#S"5;" \
    "\n\tmadc.hi.u64    "#S"6, 0x64774B84F38512BFU, "#Q"3, "#S"6;" \
\
    "\n\tmad.lo.u64.cc  "#S"4, 0x64774B84F38512BFU, "#Q"2, "#S"4;" \
    "\n\tmadc.hi.u64.cc "#S"5, 0x64774B84F38512BFU, "#Q"2, "#S"5;" \
    "\n\tmadc.lo.u64    "#S"6, 0x64774B84F38512BFU, "#Q"4, "#S"6;" \
\
    /* m2 */ \
\
    "\n\tmul.lo.u64     "#S"2, 0x6730D2A0F6B0F624U, "#Q"1       ;" \
    "\n\tmad.hi.u64.cc  "#S"3, 0x6730D2A0F6B0F624U, "#Q"1, "#S"3;" \
    "\n\tmadc.lo.u64.cc "#S"4, 0x6730D2A0F6B0F624U, "#Q"3, "#S"4;" \
    "\n\tmadc.hi.u64.cc "#S"5, 0x6730D2A0F6B0F624U, "#Q"3, "#S"5;" \
    "\n\tmadc.lo.u64    "#S"6, 0x6730D2A0F6B0F624U, "#Q"5, "#S"6;" \
\
    "\n\tmad.lo.u64.cc  "#S"3, 0x6730D2A0F6B0F624U, "#Q"2, "#S"3;" \
    "\n\tmadc.hi.u64.cc "#S"4, 0x6730D2A0F6B0F624U, "#Q"2, "#S"4;" \
    "\n\tmadc.lo.u64.cc "#S"5, 0x6730D2A0F6B0F624U, "#Q"4, "#S"5;" \
    "\n\tmadc.hi.u64    "#S"6, 0x6730D2A0F6B0F624U, "#Q"4, "#S"6;" \
\
    /* m1 */ \
\
    "\n\tmul.lo.u64     "#S"1, 0x1EABFFFEB153FFFFU, "#Q"1       ;" \
    "\n\tmad.hi.u64.cc  "#S"2, 0x1EABFFFEB153FFFFU, "#Q"1, "#S"2;" \
    "\n\tmadc.lo.u64.cc "#S"3, 0x1EABFFFEB153FFFFU, "#Q"3, "#S"3;" \
    "\n\tmadc.hi.u64.cc "#S"4, 0x1EABFFFEB153FFFFU, "#Q"3, "#S"4;" \
    "\n\tmadc.lo.u64.cc "#S"5, 0x1EABFFFEB153FFFFU, "#Q"5, "#S"5;" \
    "\n\tmadc.hi.u64    "#S"6, 0x1EABFFFEB153FFFFU, "#Q"5, "#S"6;" \
\
    "\n\tmad.lo.u64.cc  "#S"2, 0x1EABFFFEB153FFFFU, "#Q"2, "#S"2;" \
    "\n\tmadc.hi.u64.cc "#S"3, 0x1EABFFFEB153FFFFU, "#Q"2, "#S"3;" \
    "\n\tmadc.lo.u64.cc "#S"4, 0x1EABFFFEB153FFFFU, "#Q"4, "#S"4;" \
    "\n\tmadc.hi.u64.cc "#S"5, 0x1EABFFFEB153FFFFU, "#Q"4, "#S"5;" \
    "\n\tmadc.lo.u64    "#S"6, 0x1EABFFFEB153FFFFU, "#Q"6, "#S"6;" \
\
    /* m0 */ \
\
    "\n\tmul.lo.u64     "#S"0, 0xB9FEFFFFFFFFAAABU, "#Q"1       ;" \
    "\n\tmad.hi.u64.cc  "#S"1, 0xB9FEFFFFFFFFAAABU, "#Q"1, "#S"1;" \
    "\n\tmadc.lo.u64.cc "#S"2, 0xB9FEFFFFFFFFAAABU, "#Q"3, "#S"2;" \
    "\n\tmadc.hi.u64.cc "#S"3, 0xB9FEFFFFFFFFAAABU, "#Q"3, "#S"3;" \
    "\n\tmadc.lo.u64.cc "#S"4, 0xB9FEFFFFFFFFAAABU, "#Q"5, "#S"4;" \
    "\n\tmadc.hi.u64.cc "#S"5, 0xB9FEFFFFFFFFAAABU, "#Q"5, "#S"5;" \
    "\n\tmadc.lo.u64    "#S"6, 0xB9FEFFFFFFFFAAABU, "#Q"7, "#S"6;" \
\
    "\n\tmad.lo.u64.cc  "#S"1, 0xB9FEFFFFFFFFAAABU, "#Q"2, "#S"1;" \
    "\n\tmadc.hi.u64.cc "#S"2, 0xB9FEFFFFFFFFAAABU, "#Q"2, "#S"2;" \
    "\n\tmadc.lo.u64.cc "#S"3, 0xB9FEFFFFFFFFAAABU, "#Q"4, "#S"3;" \
    "\n\tmadc.hi.u64.cc "#S"4, 0xB9FEFFFFFFFFAAABU, "#Q"4, "#S"4;" \
    "\n\tmadc.lo.u64.cc "#S"5, 0xB9FEFFFFFFFFAAABU, "#Q"6, "#S"5;" \
    "\n\tmadc.hi.u64    "#S"6, 0xB9FEFFFFFFFFAAABU, "#Q"6, "#S"6;" \
\
    /* r = r1 - r2 */ \
    /*  r1 is in u */ \
    /*  r2 is in r */ \
\
    /* z = r1 - r2 */ \
\
    "\n\tsub.u64.cc  "#Z"0, "#Z"0, "#S"0;" \
    "\n\tsubc.u64.cc "#Z"1, "#Z"1, "#S"1;" \
    "\n\tsubc.u64.cc "#Z"2, "#Z"2, "#S"2;" \
    "\n\tsubc.u64.cc "#Z"3, "#Z"3, "#S"3;" \
    "\n\tsubc.u64.cc "#Z"4, "#Z"4, "#S"4;" \
    "\n\tsubc.u64    "#Z"5, "#Z"5, "#S"5;"

__forceinline__
__device__ void fp_reduce12(
    uint64_t &z0,
    uint64_t &z1,
    uint64_t &z2,
    uint64_t &z3,
    uint64_t &z4,
    uint64_t &z5,
    uint64_t z6,
    uint64_t z7,
    uint64_t z8,
    uint64_t z9,
    uint64_t za,
    uint64_t zb) {

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 z<10>, za, zb, q<8>, r<7>;"

    "\n\tmov.u64 z0,  %0;"
    "\n\tmov.u64 z1,  %1;"
    "\n\tmov.u64 z2,  %2;"
    "\n\tmov.u64 z3,  %3;"
    "\n\tmov.u64 z4,  %4;"
    "\n\tmov.u64 z5,  %5;"
    "\n\tmov.u64 z6,  %6;"
    "\n\tmov.u64 z7,  %7;"
    "\n\tmov.u64 z8,  %8;"
    "\n\tmov.u64 z9,  %9;"
    "\n\tmov.u64 za, %10;"
    "\n\tmov.u64 zb, %11;"

FP_REDUCE12(z, q, r)

    "\n\tmov.u64  %0,  z0;"
    "\n\tmov.u64  %1,  z1;"
    "\n\tmov.u64  %2,  z2;"
    "\n\tmov.u64  %3,  z3;"
    "\n\tmov.u64  %4,  z4;"
    "\n\tmov.u64  %5,  z5;"

    "\n\t}"
    :
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3), "+l"(z4), "+l"(z5)
    :
    "l"(z6), "l"(z7), "l"(z8), "l"(z9), "l"(za), "l"(zb)
    );
}

#endif
// vim: ts=4 et sw=4 si
