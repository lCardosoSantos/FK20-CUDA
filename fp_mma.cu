// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fp_mul.cuh"
#include "fp_reduce12.cuh"

__device__ void fp_mma(fp_t &z, const fp_t &v, const fp_t &w, const fp_t &x, const fp_t &y) {
    uint64_t
        v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3], v4 = v[4], v5 = v[5],
        w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5],
        x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5],
        y0 = y[0], y1 = y[1], y2 = y[2], y3 = y[3], y4 = y[4], y5 = y[5],
        z0, z1, z2, z3, z4, z5;

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 v<6>, w<6>, x<6>, y<6>;"
    "\n\t.reg .u64 u<10>, ua, ub;"
    "\n\t.reg .u64 q<8>;"
    "\n\t.reg .u64 r<7>;"
    "\n\t.reg .u32 c;"
    "\n\t.reg .pred cp;"

    "\n\tmov.u64 v0,  %6;"
    "\n\tmov.u64 v1,  %7;"
    "\n\tmov.u64 v2,  %8;"
    "\n\tmov.u64 v3,  %9;"
    "\n\tmov.u64 v4, %10;"
    "\n\tmov.u64 v5, %11;"

    "\n\tmov.u64 w0, %12;"
    "\n\tmov.u64 w1, %13;"
    "\n\tmov.u64 w2, %14;"
    "\n\tmov.u64 w3, %15;"
    "\n\tmov.u64 w4, %16;"
    "\n\tmov.u64 w5, %17;"

    "\n\tmov.u64 x0, %18;"
    "\n\tmov.u64 x1, %19;"
    "\n\tmov.u64 x2, %20;"
    "\n\tmov.u64 x3, %21;"
    "\n\tmov.u64 x4, %22;"
    "\n\tmov.u64 x5, %23;"

    "\n\tmov.u64 y0, %24;"
    "\n\tmov.u64 y1, %25;"
    "\n\tmov.u64 y2, %26;"
    "\n\tmov.u64 y3, %27;"
    "\n\tmov.u64 y4, %28;"
    "\n\tmov.u64 y5, %29;"

FP_MUL(u, v, w)

    "\n\tmov.u64 v0, u0;"
    "\n\tmov.u64 v1, u1;"
    "\n\tmov.u64 v2, u2;"
    "\n\tmov.u64 v3, u3;"
    "\n\tmov.u64 v4, u4;"
    "\n\tmov.u64 v5, u5;"

    "\n\tmov.u64 w0, u6;"
    "\n\tmov.u64 w1, u7;"
    "\n\tmov.u64 w2, u8;"
    "\n\tmov.u64 w3, u9;"
    "\n\tmov.u64 w4, ua;"
    "\n\tmov.u64 w5, ub;"

FP_MUL(u, x, y)

    // Double-width addition

    "\n\tadd.u64.cc  u0, u0, v0;"
    "\n\taddc.u64.cc u1, u1, v1;"
    "\n\taddc.u64.cc u2, u2, v2;"
    "\n\taddc.u64.cc u3, u3, v3;"
    "\n\taddc.u64.cc u4, u4, v4;"
    "\n\taddc.u64.cc u5, u5, v5;"
    "\n\taddc.u64.cc u6, u6, w0;"
    "\n\taddc.u64.cc u7, u7, w1;"
    "\n\taddc.u64.cc u8, u8, w2;"
    "\n\taddc.u64.cc u9, u9, w3;"
    "\n\taddc.u64.cc ua, ua, w4;"
    "\n\taddc.u64.cc ub, ub, w5;"
    "\n\taddc.u32 c, 0, 0;"

    // Double-width reduction

    /* if u >= 2^768 then u -= mmu0 * 2^384 */

    "\n\tsetp.ge.u32 cp, c, 1;"
    "\n@cp\tsub.u64.cc  u6, u6, 0x89f6fffffffd0003U;"
    "\n@cp\tsubc.u64.cc u7, u7, 0x140bfff43bf3fffdU;"
    "\n@cp\tsubc.u64.cc u8, u8, 0xa0b767a8ac38a745U;"
    "\n@cp\tsubc.u64.cc u9, u9, 0x8831a7ac8fada8baU;"
    "\n@cp\tsubc.u64.cc ua, ua, 0xa3f8e5685da91392U;"
    "\n@cp\tsubc.u64.cc ub, ub, 0xea09a13c057f1b6cU;"
    "\n@cp\tsubc.u32    c, c, 0;"

    /* if u >= 2^768 then u -= mmu0 * 2^384 */

    "\n\tsetp.ge.u32 cp, c, 1;"
    "\n@cp\tsub.u64.cc  u6, u6, 0x89f6fffffffd0003U;"
    "\n@cp\tsubc.u64.cc u7, u7, 0x140bfff43bf3fffdU;"
    "\n@cp\tsubc.u64.cc u8, u8, 0xa0b767a8ac38a745U;"
    "\n@cp\tsubc.u64.cc u9, u9, 0x8831a7ac8fada8baU;"
    "\n@cp\tsubc.u64.cc ua, ua, 0xa3f8e5685da91392U;"
    "\n@cp\tsubc.u64.cc ub, ub, 0xea09a13c057f1b6cU;"

FP_REDUCE12(u)

    "\n\tmov.u64 %0,  u0;"
    "\n\tmov.u64 %1,  u1;"
    "\n\tmov.u64 %2,  u2;"
    "\n\tmov.u64 %3,  u3;"
    "\n\tmov.u64 %4,  u4;"
    "\n\tmov.u64 %5,  u5;"

    "\n\t}"
    :
    "=l"(z0), "=l"(z1), "=l"(z2), "=l"(z3), "=l"(z4), "=l"(z5)
    :
    "l"(v0), "l"(v1), "l"(v2), "l"(v3), "l"(v4), "l"(v5),
    "l"(w0), "l"(w1), "l"(w2), "l"(w3), "l"(w4), "l"(w5),
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5),
    "l"(y0), "l"(y1), "l"(y2), "l"(y3), "l"(y4), "l"(y5)
    ); 

    z[0] = z0; z[1] = z1; z[2] = z2; z[3] = z3; z[4] = z4; z[5] = z5;
}

// vim: ts=4 et sw=4 si
