// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "g1.cuh"
#include "fp.cuh"
#include "fp_x2.cuh"
#include "fp_x3.cuh"
#include "fp_x8.cuh"
#include "fp_x12.cuh"
#include "fp_add.cuh"
#include "fp_sub.cuh"
#include "fp_sqr.cuh"
#include "fp_mul.cuh"
#include "fp_reduce12.cuh"

__device__ void g1p_dbl(g1p_t &p) {

    uint64_t
        x0 = p.x[0], x1 = p.x[1], x2 = p.x[2], x3 = p.x[3], x4 = p.x[4], x5 = p.x[5],
        y0 = p.y[0], y1 = p.y[1], y2 = p.y[2], y3 = p.y[3], y4 = p.y[4], y5 = p.y[5],
        z0 = p.z[0], z1 = p.z[1], z2 = p.z[2], z3 = p.z[3], z4 = p.z[4], z5 = p.z[5];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 v<6>, w<6>, x<6>, y<6>, z<6>;"
    "\n\t.reg .u64 u<10>, ua, ub;"
    "\n\t.reg .u64 q<8>;"
    "\n\t.reg .u64 r<7>;"
    "\n\t.reg .u64 t<6>;"
    "\n\t.reg .u32 t6;"
    "\n\t.reg .pred cp;"

    "\n\tmov.u64 x0,  %0;"
    "\n\tmov.u64 x1,  %1;"
    "\n\tmov.u64 x2,  %2;"
    "\n\tmov.u64 x3,  %3;"
    "\n\tmov.u64 x4,  %4;"
    "\n\tmov.u64 x5,  %5;"

    "\n\tmov.u64 y0,  %6;"
    "\n\tmov.u64 y1,  %7;"
    "\n\tmov.u64 y2,  %8;"
    "\n\tmov.u64 y3,  %9;"
    "\n\tmov.u64 y4, %10;"
    "\n\tmov.u64 y5, %11;"

    "\n\tmov.u64 z0, %12;"
    "\n\tmov.u64 z1, %13;"
    "\n\tmov.u64 z2, %14;"
    "\n\tmov.u64 z3, %15;"
    "\n\tmov.u64 z4, %16;"
    "\n\tmov.u64 z5, %17;"

    FP_MUL(x, y)
    FP_REDUCE12()

    "\n\tmov.u64 x0,  u0;"
    "\n\tmov.u64 x1,  u1;"
    "\n\tmov.u64 x2,  u2;"
    "\n\tmov.u64 x3,  u3;"
    "\n\tmov.u64 x4,  u4;"
    "\n\tmov.u64 x5,  u5;"

    FP_SQR(z)
    FP_REDUCE12()

    "\n\tmov.u64 v0,  u0;"
    "\n\tmov.u64 v1,  u1;"
    "\n\tmov.u64 v2,  u2;"
    "\n\tmov.u64 v3,  u3;"
    "\n\tmov.u64 v4,  u4;"
    "\n\tmov.u64 v5,  u5;"

    FP_X12(v, v)

    FP_MUL(z, y)
    FP_REDUCE12()

    "\n\tmov.u64 z0,  u0;"
    "\n\tmov.u64 z1,  u1;"
    "\n\tmov.u64 z2,  u2;"
    "\n\tmov.u64 z3,  u3;"
    "\n\tmov.u64 z4,  u4;"
    "\n\tmov.u64 z5,  u5;"

    FP_SQR(y)
    FP_REDUCE12()

    "\n\tmov.u64 y0,  u0;"
    "\n\tmov.u64 y1,  u1;"
    "\n\tmov.u64 y2,  u2;"
    "\n\tmov.u64 y3,  u3;"
    "\n\tmov.u64 y4,  u4;"
    "\n\tmov.u64 y5,  u5;"

    FP_X3 (w, v)

    FP_SUB(w, y, w)

    FP_MUL(x, w)
    FP_REDUCE12()

    "\n\tmov.u64 x0,  u0;"
    "\n\tmov.u64 x1,  u1;"
    "\n\tmov.u64 x2,  u2;"
    "\n\tmov.u64 x3,  u3;"
    "\n\tmov.u64 x4,  u4;"
    "\n\tmov.u64 x5,  u5;"

    FP_ADD(y, y, v)

    FP_MUL(w, y)
    FP_REDUCE12()

    "\n\tmov.u64 w0,  u0;"
    "\n\tmov.u64 w1,  u1;"
    "\n\tmov.u64 w2,  u2;"
    "\n\tmov.u64 w3,  u3;"
    "\n\tmov.u64 w4,  u4;"
    "\n\tmov.u64 w5,  u5;"

    FP_SUB(y, y, v)

    FP_X8 (y, y)

    FP_X2 (x, x)

    FP_MUL(z, y)
    FP_REDUCE12()

    "\n\tmov.u64 z0,  u0;"
    "\n\tmov.u64 z1,  u1;"
    "\n\tmov.u64 z2,  u2;"
    "\n\tmov.u64 z3,  u3;"
    "\n\tmov.u64 z4,  u4;"
    "\n\tmov.u64 z5,  u5;"

    FP_MUL(y, v)
    FP_REDUCE12()

    "\n\tmov.u64 y0,  u0;"
    "\n\tmov.u64 y1,  u1;"
    "\n\tmov.u64 y2,  u2;"
    "\n\tmov.u64 y3,  u3;"
    "\n\tmov.u64 y4,  u4;"
    "\n\tmov.u64 y5,  u5;"

    FP_ADD(y, y, w)

    "\n\tmov.u64  %0, x0;"
    "\n\tmov.u64  %1, x1;"
    "\n\tmov.u64  %2, x2;"
    "\n\tmov.u64  %3, x3;"
    "\n\tmov.u64  %4, x4;"
    "\n\tmov.u64  %5, x5;"

    "\n\tmov.u64  %6, y0;"
    "\n\tmov.u64  %7, y1;"
    "\n\tmov.u64  %8, y2;"
    "\n\tmov.u64  %9, y3;"
    "\n\tmov.u64 %10, y4;"
    "\n\tmov.u64 %11, y5;"

    "\n\tmov.u64 %12, z0;"
    "\n\tmov.u64 %13, z1;"
    "\n\tmov.u64 %14, z2;"
    "\n\tmov.u64 %15, z3;"
    "\n\tmov.u64 %16, z4;"
    "\n\tmov.u64 %17, z5;"

    "\n\t}"
    :
    "+l"(x0), "+l"(x1), "+l"(x2), "+l"(x3), "+l"(x4), "+l"(x5),
    "+l"(y0), "+l"(y1), "+l"(y2), "+l"(y3), "+l"(y4), "+l"(y5),
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3), "+l"(z4), "+l"(z5)
    );

    p.x[0] = x0; p.x[1] = x1; p.x[2] = x2; p.x[3] = x3; p.x[4] = x4; p.x[5] = x5;
    p.y[0] = y0; p.y[1] = y1; p.y[2] = y2; p.y[3] = y3; p.y[4] = y4; p.y[5] = y5;
    p.z[0] = z0; p.z[1] = z1; p.z[2] = z2; p.z[3] = z3; p.z[4] = z4; p.z[5] = z5;
}

// vim: ts=4 et sw=4 si
