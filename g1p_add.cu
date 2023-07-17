// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "fp.cuh"
#include "g1.cuh"

#include "fp_add.cuh"
#include "fp_mul.cuh"
#include "fp_sub.cuh"
#include "fp_x3.cuh"
#include "fp_x12.cuh"

// p ← p+q
// projective p and q
__device__ void g1p_add(g1p_t &p, const g1p_t &q) {

#ifndef NDEBUG
    if (!g1p_isPoint(p) || !(g1p_isPoint(q))) {
        //printf("ERROR in g1p_add(): Invalid point(s)\n");
        //g1p_print("p:", p);
        //g1p_print("q:", q);

        // return invalid point as result
        fp_zero(p.x);
        fp_zero(p.y);
        fp_zero(p.z);

        return;
    }
#endif

#if 1
    fp_t
        X1, Y1, Z1,
        X2, Y2, Z2,
        t0, t1, t2, t3;

    fp_cpy(X1, p.x);
    fp_cpy(Y1, p.y);
    fp_cpy(Z1, p.z);

    fp_cpy(X2, q.x);
    fp_cpy(Y2, q.y);
    fp_cpy(Z2, q.z);

    // Adapted from eprint 2015-1060, algorithm 7.
    // Modified to avoid overwriting inputs and remove one temp value.
    // 12 mul, 0 square, 11 add, 5 sub, 2 x12, 1 x3.

    fp_add(t0, X1, Y1); // t3
    fp_add(t1, Y1, Z1); // t8
    fp_add(t2, Z1, X1); // td

    fp_mul(X1, X1, X2); // t0
    fp_mul(Y1, Y1, Y2); // t1
    fp_mul(Z1, Z1, Z2); // t2

    fp_add(t3, X2, Y2); // t4
    fp_add(Y2, Y2, Z2); // t9
    fp_add(Z2, Z2, X2); // te

    fp_mul(X2, t3, t0); // t5
    fp_mul(Y2, Y2, t1); // ta
    fp_mul(Z2, Z2, t2); // tf

    fp_x3(t0, X1);      // ti
    fp_add(t1, Y1, Z1); // tb
    fp_add(t2, Z1, X1); // tg
    fp_x12(t3, Z1);     // tk

    fp_add(X1, X1, Y1); // t6
    fp_add(Z1, Y1, t3); // tl
    fp_sub(Y1, Y1, t3); // tm

    fp_sub(X1, X2, X1); // t7
    fp_mul(X2, X1, t0); // ts

    fp_mul(X1, X1, Y1); // tp
    fp_mul(Y1, Y1, Z1); // tr

    fp_sub(Y2, Y2, t1); // tc
    fp_mul(Z1, Z1, Y2); // tt
    fp_sub(Z2, Z2, t2); // th

    fp_x12(Z2, Z2);     // tn
    fp_mul(Y2, Y2, Z2); // to
    fp_mul(Z2, Z2, t0); // tq

    fp_sub(X1, X1, Y2); // X3
    fp_add(Y1, Y1, Z2); // Y3
    fp_add(Z1, Z1, X2); // Z3

    fp_cpy(p.x, X1);
    fp_cpy(p.y, Y1);
    fp_cpy(p.z, Z1);
#else
    uint64_t
        x0 = p.x[0], y0 = p.y[0], z0 = p.z[0],
        x1 = p.x[1], y1 = p.y[1], z1 = p.z[1],
        x2 = p.x[2], y2 = p.y[2], z2 = p.z[2],
        x3 = p.x[3], y3 = p.y[3], z3 = p.z[3],
        x4 = p.x[4], y4 = p.y[4], z4 = p.z[4],
        x5 = p.x[5], y5 = p.y[5], z5 = p.z[5],

        u0 = q.x[0], v0 = q.y[0], w0 = q.z[0],
        u1 = q.x[1], v1 = q.y[1], w1 = q.z[1],
        u2 = q.x[2], v2 = q.y[2], w2 = q.z[2],
        u3 = q.x[3], v3 = q.y[3], w3 = q.z[3],
        u4 = q.x[4], v4 = q.y[4], w4 = q.z[4],
        u5 = q.x[5], v5 = q.y[5], w5 = q.z[5];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 X1<10>, X1a, X1b;"
    "\n\t.reg .u64 X2<10>, X2a, X2b;"
    "\n\t.reg .u64 Y1<10>, Y1a, Y1b;"
    "\n\t.reg .u64 Y2<10>, Y2a, Y2b;"
    "\n\t.reg .u64 Z1<10>, Z1a, Z1b;"
    "\n\t.reg .u64 Z2<10>, Z2a, Z2b;"
    "\n\t.reg .u64 t0<6>, t1<6>, t2<6>, t3<6>;"

    "\n\t.reg .u64 t<6>;"
    "\n\t.reg .u32 z6;"
    "\n\t.reg .pred ne, gt;"

    "\n\tmov.u64 X10,  %0;"
    "\n\tmov.u64 X11,  %1;"
    "\n\tmov.u64 X12,  %2;"
    "\n\tmov.u64 X13,  %3;"
    "\n\tmov.u64 X14,  %4;"
    "\n\tmov.u64 X15,  %5;"

    "\n\tmov.u64 Y10,  %6;"
    "\n\tmov.u64 Y11,  %7;"
    "\n\tmov.u64 Y12,  %8;"
    "\n\tmov.u64 Y13,  %9;"
    "\n\tmov.u64 Y14, %10;"
    "\n\tmov.u64 Y15, %11;"

    "\n\tmov.u64 Z10, %12;"
    "\n\tmov.u64 Z11, %13;"
    "\n\tmov.u64 Z12, %14;"
    "\n\tmov.u64 Z13, %15;"
    "\n\tmov.u64 Z14, %16;"
    "\n\tmov.u64 Z15, %17;"

    "\n\tmov.u64 X20, %18;"
    "\n\tmov.u64 X21, %19;"
    "\n\tmov.u64 X22, %20;"
    "\n\tmov.u64 X23, %21;"
    "\n\tmov.u64 X24, %22;"
    "\n\tmov.u64 X25, %23;"

    "\n\tmov.u64 Y20, %24;"
    "\n\tmov.u64 Y21, %25;"
    "\n\tmov.u64 Y22, %26;"
    "\n\tmov.u64 Y23, %27;"
    "\n\tmov.u64 Y24, %28;"
    "\n\tmov.u64 Y25, %29;"

    "\n\tmov.u64 Z20, %30;"
    "\n\tmov.u64 Z21, %31;"
    "\n\tmov.u64 Z22, %32;"
    "\n\tmov.u64 Z23, %33;"
    "\n\tmov.u64 Z24, %34;"
    "\n\tmov.u64 Z25, %35;"

FP_ADD(t0, X1, Y1) // t3
FP_ADD(t1, Y1, Z1) // t8
FP_ADD(t2, Z1, X1) // td

FP_MUL(X1, X1, X2) // t0
FP_REDUCE12(X1)
FP_MUL(Y1, Y1, Y2) // t1
FP_REDUCE12(Y1)
FP_MUL(Z1, Z1, Z2) // t2
FP_REDUCE12(Z1)

FP_ADD(t3, X2, Y2) // t4
FP_ADD(Y2, Y2, Z2) // t9
FP_ADD(Z2, Z2, X2) // te

FP_MUL(X2, t3, t0) // t5
FP_REDUCE12(X2)
FP_MUL(Y2, Y2, t1) // ta
FP_REDUCE12(Y2)
FP_MUL(Z2, Z2, t2) // tf
FP_REDUCE12(Z2)

FP_X3(t0, X1)      // ti
FP_ADD(t1, Y1, Z1) // tb
FP_ADD(t2, Z1, X1) // tg
FP_X12(t3, Z1)     // tk

FP_ADD(X1, X1, Y1) // t6
FP_ADD(Z1, Y1, t3) // tl
FP_SUB(Y1, Y1, t3) // tm

FP_SUB(X1, X2, X1) // t7
FP_MUL(X2, X1, t0) // ts
FP_REDUCE12(X2)

FP_MUL(X1, X1, Y1) // tp
FP_REDUCE12(X1)
FP_MUL(Y1, Y1, Z1) // tr
FP_REDUCE12(Y1)

FP_SUB(Y2, Y2, t1) // tc
FP_MUL(Z1, Z1, Y2) // tt
FP_REDUCE12(Z1)
FP_SUB(Z2, Z2, t2) // th

FP_X12(Z2, Z2)     // tn
FP_MUL(Y2, Y2, Z2) // to
FP_REDUCE12(Y2)
FP_MUL(Z2, Z2, t0) // tq
FP_REDUCE12(Z2)

FP_SUB(X1, X1, Y2) // X3
FP_ADD(Y1, Y1, Z2) // Y3
FP_ADD(Z1, Z1, X2) // Z3

    "\n\tmov.u64  %0,  X10;"
    "\n\tmov.u64  %1,  X11;"
    "\n\tmov.u64  %2,  X12;"
    "\n\tmov.u64  %3,  X13;"
    "\n\tmov.u64  %4,  X14;"
    "\n\tmov.u64  %5,  X15;"

    "\n\tmov.u64  %6,  Y10;"
    "\n\tmov.u64  %7,  Y11;"
    "\n\tmov.u64  %8,  Y12;"
    "\n\tmov.u64  %9,  Y13;"
    "\n\tmov.u64 %10,  Y14;"
    "\n\tmov.u64 %11,  Y15;"

    "\n\tmov.u64 %12,  Z10;"
    "\n\tmov.u64 %13,  Z11;"
    "\n\tmov.u64 %14,  Z12;"
    "\n\tmov.u64 %15,  Z13;"
    "\n\tmov.u64 %16,  Z14;"
    "\n\tmov.u64 %17,  Z15;"

    "\n\t}"
    :
    "+l"(x0), "+l"(x1), "+l"(x2), "+l"(x3), "+l"(x4), "+l"(x5),
    "+l"(y0), "+l"(y1), "+l"(y2), "+l"(y3), "+l"(y4), "+l"(y5),
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3), "+l"(z4), "+l"(z5),
    :
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5),
    "l"(y0), "l"(y1), "l"(y2), "l"(y3), "l"(y4), "l"(y5),
    "l"(z0), "l"(z1), "l"(z2), "l"(z3), "l"(z4), "l"(z5),
    );

    p.x[0] = x0, p.x[1] = x1, p.x[2] = x2, p.x[3] = x3, p.x[4] = x4, p.x[5] = x5;
    p.y[0] = y0, p.y[1] = y1, p.y[2] = y2, p.y[3] = y3, p.y[4] = y4, p.y[5] = y5;
    p.z[0] = z0, p.z[1] = z1, p.z[2] = z2, p.z[3] = z3, p.z[4] = z4, p.z[5] = z5;
#endif
}

// vim: ts=4 et sw=4 si
