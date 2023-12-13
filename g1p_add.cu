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

// Temporaries
#define t0 t00, t01, t02, t03, t04, t05
#define t1 t10, t11, t12, t13, t14, t15
#define t2 t20, t21, t22, t23, t24, t25
#define t3 t30, t31, t32, t33, t34, t35
#define t4 t40, t41, t42, t43, t44, t45

#define X1 X10, X11, X12, X13, X14, X15
#define Y1 Y10, Y11, Y12, Y13, Y14, Y15
#define Z1 Z10, Z11, Z12, Z13, Z14, Z15

#define X2 X20, X21, X22, X23, X24, X25
#define Y2 Y20, Y21, Y22, Y23, Y24, Y25
#define Z2 Z20, Z21, Z22, Z23, Z24, Z25

#define X3 X30, X31, X32, X33, X34, X35
#define Y3 Y30, Y31, Y32, Y33, Y34, Y35
#define Z3 Z30, Z31, Z32, Z33, Z34, Z35


/**
 * @brief Computes the projective sum of the projective point p and the affine point q.
 *
 *  p ← p+q
 *
 * @param[in, out] p Projective G1 point
 * @param[in] q Affine G1 point
 * @return void
 *
 */
__device__ void g1p_add(g1p_t &p, const g1a_t &q) {

    uint64_t
        A,
        X1, Y1, Z1,
        X2, Y2,
        X3, Y3, Z3,
        t0, t1, t2, t3, t4;

    fp_cpy(X1, PX);
    fp_cpy(Y1, PY);
    fp_cpy(Z1, PZ);

    fp_cpy(X2, QX);
    fp_cpy(Y2, QY);

    // Adapted from eprint 2015-1060, algorithm 8.
    // 11 mul, 0 square, 8 add, 3 sub, 2 x12, 1 x3.

    fp_mul(A, X1, X2);      // 1
    fp_reduce12(AL, A);
    fp_cpy(t0, AL);

    fp_mul(A, Y1, Y2);      // 2
    fp_reduce12(AL, A);
    fp_cpy(t1, AL);

    fp_add(t3, X2, Y2);     // 3
    fp_add(t4, X1, Y1);     // 4

    fp_mul(A, t3, t4);      // 5
    fp_reduce12(AL, A);
    fp_cpy(t3, AL);

    fp_add(t4, t0, t1);     // 6
    fp_sub(t3, t3, t4);     // 7

    fp_mul(A, Y2, Z1);      // 8
    fp_reduce12(AL, A);
    fp_cpy(t4, AL);

    fp_add(t4, t4, Y1);     // 9

    fp_mul(A, X2, Z1);      // 10
    fp_reduce12(AL, A);
    fp_cpy(Y3, AL);

    fp_add(Y3, Y3, X1);     // 11
    fp_x3(t0, t0);          // 12-13
    fp_x12(t2, Z1);         // 14
    fp_add(Z3, t1, t2);     // 15
    fp_sub(t1, t1, t2);     // 16
    fp_x12(Y3, Y3);         // 17

    fp_mul(A, t4, Y3);      // 18
    fp_reduce12(AL, A);
    fp_cpy(X3, AL);

    fp_mul(A, t3, t1);      // 19
    fp_reduce12(AL, A);
    fp_cpy(t2, AL);

    fp_sub(X3, t2, X3);     // 20

    fp_mul(A, Y3, t0);      // 21
    fp_reduce12(AL, A);
    fp_cpy(Y3, AL);

    fp_mul(A, t1, Z3);      // 22
    fp_reduce12(AL, A);
    fp_cpy(t1, AL);

    fp_add(Y3, t1, Y3);     // 23

    fp_mul(A, t0, t3);      // 24
    fp_reduce12(AL, A);
    fp_cpy(t0, AL);

    fp_mul(A, Z3, t4);      // 25
    fp_reduce12(AL, A);
    fp_cpy(Z3, AL);

    fp_add(Z3, Z3, t0);     // 26

    fp_cpy(PX, X3);
    fp_cpy(PY, Y3);
    fp_cpy(PZ, Z3);
}

// vim: ts=4 et sw=4 si
