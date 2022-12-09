// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "g1.cuh"

// p ← p+q
// projective p and q
__device__ void g1p_add(uint64_t *p, const uint64_t *q) {

    uint64_t
        X1[6], Y1[6], Z1[6],
        X2[6], Y2[6], Z2[6],
        t0[6], t1[6], t2[6], t3[6];

    fp_cpy(X1, p+ 0);
    fp_cpy(Y1, p+ 6);
    fp_cpy(Z1, p+12);

    fp_cpy(X2, q+ 0);
    fp_cpy(Y2, q+ 6);
    fp_cpy(Z2, q+12);

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

    fp_cpy(p+ 0, X1);
    fp_cpy(p+ 6, Y1);
    fp_cpy(p+12, Z1);
}

// vim: ts=4 et sw=4 si
