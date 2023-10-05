// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fp.cuh"
#include "g1.cuh"

#include "fp_cpy.cuh"
#include "fp_add.cuh"
#include "fp_mul.cuh"
#include "fp_sub.cuh"
#include "fp_x3.cuh"
#include "fp_x12.cuh"
#include "fp_reduce12.cuh"

// I/O

#define PX p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5]
#define PY p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5]
#define PZ p.z[0], p.z[1], p.z[2], p.z[3], p.z[4], p.z[5]

#define QX q.x[0], q.x[1], q.x[2], q.x[3], q.x[4], q.x[5]
#define QY q.y[0], q.y[1], q.y[2], q.y[3], q.y[4], q.y[5]
#define QZ q.z[0], q.z[1], q.z[2], q.z[3], q.z[4], q.z[5]

// Accumulator
#define A a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab
#define AL a0, a1, a2, a3, a4, a5

// Operands
#define B b0, b1, b2, b3, b4, b5
#define C c0, c1, c2, c3, c4, c5

// Temporaries
#define t0 t00, t01, t02, t03, t04, t05
#define t1 t10, t11, t12, t13, t14, t15
#define t2 t20, t21, t22, t23, t24, t25
#define t3 t30, t31, t32, t33, t34, t35

#define X1 X10, X11, X12, X13, X14, X15
#define Y1 Y10, Y11, Y12, Y13, Y14, Y15
#define Z1 Z10, Z11, Z12, Z13, Z14, Z15

#define X2 X20, X21, X22, X23, X24, X25
#define Y2 Y20, Y21, Y22, Y23, Y24, Y25
#define Z2 Z20, Z21, Z22, Z23, Z24, Z25


/** 
 * @brief Computes the sum of two points  q into p, using projective coordinates.
 * and stores in p.
 * 
 *  p ← p+q
 * 
 * @param[in, out] p accumulator
 * @param[in] q second operand
 * @return void 
 * 
 */
__device__ void g1p_add(g1p_t &p, const g1p_t &q) {

#if 0 //ndef NDEBUG
    if (!g1p_isPoint(p) || !(g1p_isPoint(q))) {
        //printf("ERROR in g1p_add(): Invalid point(s)\n");
        //g1p_print("p: ", p);
        //g1p_print("q: ", q);

        // return invalid point as result
        fp_zero(p.x);
        fp_zero(p.y);
        fp_zero(p.z);

        return;
    }
#endif

#if 1
    g1p_multi(1, &p, NULL, &p, &q);   // p ← p+q
#else
    uint64_t
        A, B, C,
        X1, Y1, Z1,
        X2, Y2, Z2,
        t0, t1, t2, t3;

    fp_cpy(X1, PX);
    fp_cpy(Y1, PY);
    fp_cpy(Z1, PZ);

    fp_cpy(X2, QX);
    fp_cpy(Y2, QY);
    fp_cpy(Z2, QZ);

    // Adapted from eprint 2015-1060, algorithm 7.
    // Modified to remove one temp value and avoid overwriting inputs.
    // 12 mul, 0 square, 11 add, 5 sub, 2 x12, 1 x3.

    fp_add(t0, X1, Y1); // t3
    fp_add(t1, Y1, Z1); // t8
    fp_add(t2, Z1, X1); // td

    fp_mul(A, X1, X2); // t0
    fp_reduce12(A);
    fp_cpy(X1, AL);
    fp_mul(A, Y1, Y2); // t1
    fp_reduce12(A);
    fp_cpy(Y1, AL);
    fp_mul(A, Z1, Z2); // t2
    fp_reduce12(A);
    fp_cpy(Z1, AL);

    fp_add(t3, X2, Y2); // t4
    fp_add(Y2, Y2, Z2); // t9
    fp_add(Z2, Z2, X2); // te

    fp_mul(A, t3, t0); // t5
    fp_reduce12(A);
    fp_cpy(X2, AL);
    fp_mul(A, Y2, t1); // ta
    fp_reduce12(A);
    fp_cpy(Y2, AL);
    fp_mul(A, Z2, t2); // tf
    fp_reduce12(A);
    fp_cpy(Z2, AL);

    fp_x3(t0, X1);      // ti
    fp_add(t1, Y1, Z1); // tb
    fp_add(t2, Z1, X1); // tg
    fp_x12(t3, Z1);     // tk

    fp_add(X1, X1, Y1); // t6
    fp_add(Z1, Y1, t3); // tl
    fp_sub(Y1, Y1, t3); // tm

    fp_sub(X1, X2, X1); // t7
    fp_mul(A, X1, t0); // ts
    fp_reduce12(A);
    fp_cpy(X2, AL);

    fp_mul(A, X1, Y1); // tp
    fp_reduce12(A);
    fp_cpy(X1, AL);
    fp_mul(A, Y1, Z1); // tr
    fp_reduce12(A);
    fp_cpy(Y1, AL);

    fp_sub(Y2, Y2, t1); // tc
    fp_mul(A, Z1, Y2); // tt
    fp_reduce12(A);
    fp_cpy(Z1, AL);
    fp_sub(Z2, Z2, t2); // th

    fp_x12(Z2, Z2);     // tn
    fp_mul(A, Y2, Z2); // to
    fp_reduce12(A);
    fp_cpy(Y2, AL);
    fp_mul(A, Z2, t0); // tq
    fp_reduce12(A);
    fp_cpy(Z2, AL);

    fp_sub(X1, X1, Y2); // X3
    fp_add(Y1, Y1, Z2); // Y3
    fp_add(Z1, Z1, X2); // Z3

    fp_cpy(PX, X1);
    fp_cpy(PY, Y1);
    fp_cpy(PZ, Z1);
#endif
}

// vim: ts=4 et sw=4 si
