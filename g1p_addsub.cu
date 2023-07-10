// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "fp.cuh"
#include "g1.cuh"

// p,q ← p+q,p-q
// projective p and q
__device__ void g1p_addsub(g1p_t &p, g1p_t &q) {

#ifndef NDEBUG
    if (!g1p_isPoint(p) || !(g1p_isPoint(q))) {
        printf("ERROR in g1p_addsub(): Invalid point(s)\n");
        g1p_print("p:", p);
        g1p_print("q:", q);

        // return invalid points as result
        fp_zero(p.x); fp_zero(q.x);
        fp_zero(p.y); fp_zero(q.y);
        fp_zero(p.z); fp_zero(q.z);

        return;
    }
#endif


    fp_t
        &X1 = p.x, &Y1 = p.y, &Z1 = p.z,
        &X2 = q.x, &Y2 = q.y, &Z2 = q.z,
        t0, t1, t2, t3;

    //printf("X1 = "); fp_print(X1);
    //printf("Y1 = "); fp_print(Y1);
    //printf("Z1 = "); fp_print(Z1);

    //printf("X2 = "); fp_print(X2);
    //printf("Y2 = "); fp_print(Y2);
    //printf("Z2 = "); fp_print(Z2);
    //printf("\n");

    fp_mul(t0, X1, X2); // t0
    //printf("t0 = "); fp_print(t0);

    fp_add(t3, X1, Y1); // t3
    //printf("t3 = "); fp_print(t3);

    fp_add(X1, X1, Z1); // td
    //printf("td = "); fp_print(X1);

    fp_add(t2, X2, Z2); // te
    //printf("te = "); fp_print(t2);

    fp_mul(X1, X1, t2); // tf
    //printf("tf = "); fp_print(X1);


    fp_add(t1, Y2, Z2); // t9
    //printf("t9 = "); fp_print(t1);

    fp_mul(t2, Z1, Z2); // t2
    //printf("t2 = "); fp_print(t2);

    fp_add(Z1, Z1, Y1); // t8
    //printf("t8 = "); fp_print(Z1);

    fp_sub(Z2, Z2, Y2); // T9
    //printf("T9 = "); fp_print(Z2);
    fp_mul(Z2, Z2, Z1); // Ta
    //printf("Ta = "); fp_print(Z2);
    fp_mul(Z1, Z1, t1); // ta

    //printf("ta = "); fp_print(Z1);
    fp_sub(Z1, Z1, t2); // tc

    fp_add(t1, X2, Y2); // t4
    //printf("t4 = "); fp_print(t1);

    fp_mul(t1, t1, t3); // t5
    //printf("t5 = "); fp_print(t1);

    fp_sub(X1, X1, t2); // (th)
    fp_sub(X2, X2, Y2); // T4
    //printf("T4 = "); fp_print(X2);
    fp_mul(X2, X2, t3); // T5
    //printf("T5 = "); fp_print(X2);
    fp_sub(X2, X2, t0); // T7

    fp_mul(Y1, Y1, Y2); // t1
    //printf("t1 = "); fp_print(Y1);

    fp_x12(Y2, t2);     // tk
    //printf("tk = "); fp_print(Y2);

    fp_add(Z2, Z2, Y1); // Tc
    fp_sub(Z2, Z2, t2); // Tc
    //printf("Tc = "); fp_print(Z2);

    fp_sub(t2, t1, t0); // (t7)
    fp_sub(t2, t2, Y1); // t7
    //printf("t7 = "); fp_print(t2);

    fp_add(X2, X2, Y1); // T7
    //printf("T7 = "); fp_print(X2);

    fp_sub(Z1, Z1, Y1); // tc
    //printf("tc = "); fp_print(Z1);

    fp_sub(X1, X1, t0); // th
    //printf("th = "); fp_print(X1);

    fp_x3(t0, t0);      // ti
    //printf("ti = "); fp_print(t0);

    fp_x12(X1, X1);     // tn
    //printf("tn = "); fp_print(X1);


    fp_add(t3, Y2, Y1); // tl
    //printf("tl = "); fp_print(t3);

    fp_sub(Y1, Y1, Y2); // tm
    //printf("tm = "); fp_print(Y1);


    // Active (tag/var) = t7/t2, tc/Z1, ti/t0, tl/t3, tm/Y1, tn/X1, T7/X2, Tc/Z2
    // Available (var) = t1, Y2

    fp_cpy(t1, X2); // T7

    fp_mma(X2, t1, t3, Z2, X1); // T7, -Tm=tl, Tc, tn
    fp_neg(X2, X2); // X2

    fp_neg(Z2, Z2); // -Tc

    fp_mma(Z2, t1, t0, Z2, Y1); // T7, ti, -Tc, tm

    fp_mma(Y2, t0, X1, t3, Y1); // ti, tn, tl, tm

    fp_neg(X1, X1); // -tn
    fp_mma(X1, t2, Y1, Z1, X1); // t7, tm, tc, -tn
    fp_mma(Z1, t2, t0, Z1, t3); // t7, ti, tc, tl

    fp_cpy(Y1, Y2);

    //printf("X3 = "); fp_print(X1);
    //printf("Y3 = "); fp_print(Y1);
    //printf("Z3 = "); fp_print(Z1);

    //printf("X4 = "); fp_print(X2);
    //printf("Y4 = "); fp_print(Y2);
    //printf("Z4 = "); fp_print(Z2);
}

// vim: ts=4 et sw=4 si
