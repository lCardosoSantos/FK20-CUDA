// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

// #include <stdio.h>

#include "fp.cuh"
#include "g1.cuh"

// p ← p-q
// projective p and q
__device__ void g1p_sub(g1p_t &p, const g1p_t &q) {

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

    if (!g1p_isInf(q))
        fp_neg(Y2, Y2);

    // Adapted from eprint 2015-1060, algorithm 7.
    // Modified to avoid overwriting inputs and remove one temp value.
    // 12 mul, 0 square, 11 add, 5 sub, 2 x12, 1 x3.

    fp_add(t0, X1, Y1); // t3

    //printf("T3 = "); fp_print(t0);

    fp_add(t1, Y1, Z1); // t8

    //printf("T8 = "); fp_print(t1);

    fp_add(t2, Z1, X1); // td

    //printf("Td = "); fp_print(t2);


    fp_mul(X1, X1, X2); // t0

    //printf("T0 = "); fp_print(X1);

    fp_mul(Y1, Y1, Y2); // t1

    //printf("T1 = "); fp_print(Y1);

    fp_mul(Z1, Z1, Z2); // t2

    //printf("T2 = "); fp_print(Z1);


    fp_add(t3, X2, Y2); // t4

    //printf("T4 = "); fp_print(t3);

    fp_add(Y2, Y2, Z2); // t9

    //printf("T9 = "); fp_print(Y2);

    fp_add(Z2, Z2, X2); // te

    //printf("Te = "); fp_print(Z2);


    fp_mul(X2, t3, t0); // t5

    //printf("T5 = "); fp_print(X2);

    fp_mul(Y2, Y2, t1); // ta

    //printf("Ta = "); fp_print(Y2);

    fp_mul(Z2, Z2, t2); // tf

    //printf("Tf = "); fp_print(Z2);


    fp_x3(t0, X1);      // ti

    //printf("Ti = "); fp_print(t0);

    fp_add(t1, Y1, Z1); // tb

    //printf("Tb = "); fp_print(t1);

    fp_add(t2, Z1, X1); // tg

    //printf("Tg = "); fp_print(t2);

    fp_x12(t3, Z1);     // tk

    //printf("Tk = "); fp_print(t3);


    fp_add(X1, X1, Y1); // t6

    //printf("T6 = "); fp_print(X1);

    fp_add(Z1, Y1, t3); // tl

    //printf("Tl = "); fp_print(Z1);

    fp_sub(Y1, Y1, t3); // tm

    //printf("Tm = "); fp_print(Y1);


    fp_sub(X1, X2, X1); // t7

    //printf("T7 = "); fp_print(X1);

    fp_mul(X2, X1, t0); // ts

    //printf("Ts = "); fp_print(X2);


    fp_mul(X1, X1, Y1); // tp

    //printf("Tp = "); fp_print(X1);

    fp_mul(Y1, Y1, Z1); // tr

    //printf("Tr = "); fp_print(Y1);


    fp_sub(Y2, Y2, t1); // tc

    //printf("Tc = "); fp_print(Y2);

    fp_mul(Z1, Z1, Y2); // tt

    //printf("Tt = "); fp_print(Z1);

    fp_sub(Z2, Z2, t2); // th

    //printf("Th = "); fp_print(Z2);


    fp_x12(Z2, Z2);     // tn

    //printf("Tn = "); fp_print(Z2);

    fp_mul(Y2, Y2, Z2); // to

    //printf("To = "); fp_print(Y2);

    fp_mul(Z2, Z2, t0); // tq

    //printf("Tq = "); fp_print(Z2);


    fp_sub(X1, X1, Y2); // X3
    fp_add(Y1, Y1, Z2); // Y3
    fp_add(Z1, Z1, X2); // Z3

    //printf("X4 = "); fp_print(X1);
    //printf("Y4 = "); fp_print(Y1);
    //printf("Z4 = "); fp_print(Z1);

    fp_cpy(p.x, X1);
    fp_cpy(p.y, Y1);
    fp_cpy(p.z, Z1);
}

// vim: ts=4 et sw=4 si
