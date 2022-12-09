// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"

__device__ void fr_mul(uint64_t *z, const uint64_t *x) {
    uint64_t
        z0 = z[0], z1 = z[1], z2 = z[2], z3 = z[3],
        x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 u<8>;"
    "\n\t.reg .u64 q<6>;"
    "\n\t.reg .u64 r<5>;"
    "\n\t.reg .pred cp;"

    // mul

    "\n\tmul.lo.u64     u1, %0, %5    ; mul.hi.u64     u2, %0, %5    ;"
    "\n\tmul.lo.u64     u3, %0, %7    ; mul.hi.u64     u4, %0, %7    ;"

    "\n\tmul.lo.u64     u0, %0, %4    ; mad.hi.u64.cc  u1, %0, %4, u1;"
    "\n\tmadc.lo.u64.cc u2, %0, %6, u2; madc.hi.u64.cc u3, %0, %6, u3;"
    "\n\taddc.u64       u4,  0,  u4;"


    "\n\tmad.lo.u64.cc  u2, %1, %5, u2; madc.hi.u64.cc u3, %1, %5, u3;"
    "\n\tmadc.lo.u64.cc u4, %1, %7, u4; madc.hi.u64    u5, %1, %7,  0;"

    "\n\tmad.lo.u64.cc  u1, %1, %4, u1; madc.hi.u64.cc u2, %1, %4, u2;"
    "\n\tmadc.lo.u64.cc u3, %1, %6, u3; madc.hi.u64.cc u4, %1, %6, u4;"
    "\n\taddc.u64       u5,  0,  u5;"


    "\n\tmad.lo.u64.cc  u3, %2, %5, u3; madc.hi.u64.cc u4, %2, %5, u4;"
    "\n\tmadc.lo.u64.cc u5, %2, %7, u5; madc.hi.u64    u6, %2, %7,  0;"

    "\n\tmad.lo.u64.cc  u2, %2, %4, u2; madc.hi.u64.cc u3, %2, %4, u3;"
    "\n\tmadc.lo.u64.cc u4, %2, %6, u4; madc.hi.u64.cc u5, %2, %6, u5;"
    "\n\taddc.u64       u6,  0,  u6;"


    "\n\tmad.lo.u64.cc  u4, %3, %5, u4; madc.hi.u64.cc u5, %3, %5, u5;"
    "\n\tmadc.lo.u64.cc u6, %3, %7, u6; madc.hi.u64    u7, %3, %7,  0;"

    "\n\tmad.lo.u64.cc  u3, %3, %4, u3; madc.hi.u64.cc u4, %3, %4, u4;"
    "\n\tmadc.lo.u64.cc u5, %3, %6, u5; madc.hi.u64.cc u6, %3, %6, u6;"
    "\n\taddc.u64       u7,  0,  u7;"

    // reduce8

    // q2 = q1 * mu; q3 = q2 / 2^320

    // mu0

    "\n\tmul.hi.u64     q0, 0x42737A020C0D6393U, u6;"

    "\n\tmad.lo.u64.cc  q0, 0x42737A020C0D6393U, u7, q0;"
    "\n\tmadc.hi.u64    q1, 0x42737A020C0D6393U, u7,  0;"

    // mu1

    "\n\tmad.hi.u64.cc  q0, 0x65043EB4BE4BAD71U, u5, q0;"
    "\n\tmadc.lo.u64.cc q1, 0x65043EB4BE4BAD71U, u7, q1;"
    "\n\tmadc.hi.u64    q2, 0x65043EB4BE4BAD71U, u7,  0;"

    "\n\tmad.lo.u64.cc  q0, 0x65043EB4BE4BAD71U, u6, q0;"
    "\n\tmadc.hi.u64.cc q1, 0x65043EB4BE4BAD71U, u6, q1;"
    "\n\taddc.u64       q2, q2, 0;"

    // mu2

    "\n\tmad.lo.u64.cc  q0, 0x38B5DCB707E08ED3U, u5, q0;"
    "\n\tmadc.hi.u64.cc q1, 0x38B5DCB707E08ED3U, u5, q1;"
    "\n\tmadc.lo.u64.cc q2, 0x38B5DCB707E08ED3U, u7, q2;"
    "\n\tmadc.hi.u64    q3, 0x38B5DCB707E08ED3U, u7,  0;"

    "\n\tmad.hi.u64.cc  q0, 0x38B5DCB707E08ED3U, u4, q0;"
    "\n\tmadc.lo.u64.cc q1, 0x38B5DCB707E08ED3U, u6, q1;"
    "\n\tmadc.hi.u64.cc q2, 0x38B5DCB707E08ED3U, u6, q2;"
    "\n\taddc.u64       q3, q3, 0;"

    // mu3

    "\n\tmad.hi.u64.cc  q0, 0x355094EDFEDE377CU, u3, q0;"
    "\n\tmadc.lo.u64.cc q1, 0x355094EDFEDE377CU, u5, q1;"
    "\n\tmadc.hi.u64.cc q2, 0x355094EDFEDE377CU, u5, q2;"
    "\n\tmadc.lo.u64.cc q3, 0x355094EDFEDE377CU, u7, q3;"
    "\n\tmadc.hi.u64    q4, 0x355094EDFEDE377CU, u7,  0;"

    "\n\tmad.lo.u64.cc  q0, 0x355094EDFEDE377CU, u4, q0;"
    "\n\tmadc.hi.u64.cc q1, 0x355094EDFEDE377CU, u4, q1;"
    "\n\tmadc.lo.u64.cc q2, 0x355094EDFEDE377CU, u6, q2;"
    "\n\tmadc.hi.u64.cc q3, 0x355094EDFEDE377CU, u6, q3;"
    "\n\taddc.u64       q4, q4, 0;"

    // mu4

    "\n\tmad.lo.u64.cc  q0, 0x0000000000000002U, u3, q0;"
    "\n\tmadc.hi.u64.cc q1, 0x0000000000000002U, u3, q1;"
    "\n\tmadc.lo.u64.cc q2, 0x0000000000000002U, u5, q2;"
    "\n\tmadc.hi.u64.cc q3, 0x0000000000000002U, u5, q3;"
    "\n\tmadc.lo.u64.cc q4, 0x0000000000000002U, u7, q4;"
    "\n\tmadc.hi.u64    q5, 0x0000000000000002U, u7,  0;"

    "\n\tmad.hi.u64.cc  q0, 0x0000000000000002U, u2, q0;"
    "\n\tmadc.lo.u64.cc q1, 0x0000000000000002U, u4, q1;"
    "\n\tmadc.hi.u64.cc q2, 0x0000000000000002U, u4, q2;"
    "\n\tmadc.lo.u64.cc q3, 0x0000000000000002U, u6, q3;"
    "\n\tmadc.hi.u64.cc q4, 0x0000000000000002U, u6, q4;"
    "\n\taddc.u64       q5, q5, 0;"

    // r2 = q3 * m mod 2^320
    //  u contains z^2
    //  q contains q3
    //  produces r2 in r

    // m3

    "\n\tmul.lo.u64     r3, 0x73EDA753299D7D48U, q1    ;"
    "\n\tmul.hi.u64     r4, 0x73EDA753299D7D48U, q1    ;"
    "\n\tmad.lo.u64     r4, 0x73EDA753299D7D48U, q2, r4;"

    // m2

    "\n\tmul.lo.u64     r2, 0x3339D80809A1D805U, q1    ;"
    "\n\tmad.hi.u64.cc  r3, 0x3339D80809A1D805U, q1, r3;"
    "\n\tmadc.lo.u64    r4, 0x3339D80809A1D805U, q3, r4;"

    "\n\tmad.lo.u64.cc  r3, 0x3339D80809A1D805U, q2, r3;"
    "\n\tmadc.hi.u64    r4, 0x3339D80809A1D805U, q2, r4;"

    // m1

    "\n\tmul.lo.u64     r1, 0x53BDA402FFFE5BFEU, q1    ;"
    "\n\tmad.hi.u64.cc  r2, 0x53BDA402FFFE5BFEU, q1, r2;"
    "\n\tmadc.lo.u64.cc r3, 0x53BDA402FFFE5BFEU, q3, r3;"
    "\n\tmadc.hi.u64    r4, 0x53BDA402FFFE5BFEU, q3, r4;"

    "\n\tmad.lo.u64.cc  r2, 0x53BDA402FFFE5BFEU, q2, r2;"
    "\n\tmadc.hi.u64.cc r3, 0x53BDA402FFFE5BFEU, q2, r3;"
    "\n\tmadc.lo.u64    r4, 0x53BDA402FFFE5BFEU, q4, r4;"

    // m0

    "\n\tmul.lo.u64     r0, 0xFFFFFFFF00000001U, q1    ;"
    "\n\tmad.hi.u64.cc  r1, 0xFFFFFFFF00000001U, q1, r1;"
    "\n\tmadc.lo.u64.cc r2, 0xFFFFFFFF00000001U, q3, r2;"
    "\n\tmadc.hi.u64.cc r3, 0xFFFFFFFF00000001U, q3, r3;"
    "\n\tmadc.lo.u64    r4, 0xFFFFFFFF00000001U, q5, r4;"

    "\n\tmad.lo.u64.cc  r1, 0xFFFFFFFF00000001U, q2, r1;"
    "\n\tmadc.hi.u64.cc r2, 0xFFFFFFFF00000001U, q2, r2;"
    "\n\tmadc.lo.u64.cc r3, 0xFFFFFFFF00000001U, q4, r3;"
    "\n\tmadc.hi.u64    r4, 0xFFFFFFFF00000001U, q4, r4;"

    // r = r1 - r2
    //  r1 is in u
    //  r2 is in r

    // z = r1 - r2

    // Note: 0 <= z < 3m and 2m < 2^256, so z >= 2^256 => 0 < z-m < 2^256

    "\n\tsub.u64.cc  %0, u0, r0;"
    "\n\tsubc.u64.cc %1, u1, r1;"
    "\n\tsubc.u64.cc %2, u2, r2;"
    "\n\tsubc.u64.cc %3, u3, r3;"
    "\n\tsubc.u64.cc u4, u4, r4;"
    "\n\tsetp.ne.u64 cp, u4, 0;" // set predicate if z >= 2^256

    // if predicate is set then z = z - m

    "\n @cp\tsub.u64.cc  %0, %0, 0xFFFFFFFF00000001U;"
    "\n @cp\tsubc.u64.cc %1, %1, 0x53BDA402FFFE5BFEU;"
    "\n @cp\tsubc.u64.cc %2, %2, 0x3339D80809A1D805U;"
    "\n @cp\tsubc.u64    %3, %3, 0x73EDA753299D7D48U;"

    "\n\t}"
    : "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3)
    : "l"(x0), "l"(x1), "l"(x2), "l"(x3)
    ); 

    z[0] = z0; z[1] = z1; z[2] = z2; z[3] = z3;
}

// vim: ts=4 et sw=4 si
