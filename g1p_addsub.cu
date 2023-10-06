// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fp.cuh"
#include "g1.cuh"

/**
 * @brief Stores the sum and difference of p and q into p and q. p,q ← p+q,p-q
 * Projective p and q.
 * 
 * @param[in, out] p First parameter, stores p+q
 * @param[in, out] q Second parameter, stores p-q
 * @return void 
 */
__device__ void g1p_addsub(g1p_t &p, g1p_t &q) {

#if 0 //ndef NDEBUG
    if (!g1p_isPoint(p) || !(g1p_isPoint(q))) {
        //printf("ERROR in g1p_addsub(): Invalid point(s)\n");
        //g1p_print("p: ", p);
        //g1p_print("q: ", q);

        // return invalid points as result
        fp_zero(p.x); fp_zero(q.x);
        fp_zero(p.y); fp_zero(q.y);
        fp_zero(p.z); fp_zero(q.z);

        return;
    }
#endif

#if 1
    g1p_multi(-3, &p, &q, &p, &q);
#else
    fp_t
        &X1 = p.x, &Y1 = p.y, &Z1 = p.z,
        &X2 = q.x, &Y2 = q.y, &Z2 = q.z,
        t0, t1, t2, t3;

    //fp_print("X1 = ",  X1);
    //fp_print("Y1 = ",  Y1);
    //fp_print("Z1 = ",  Z1);

    //fp_print("X2 = ",  X2);
    //fp_print("Y2 = ",  Y2);
    //fp_print("Z2 = ",  Z2);
    //printf("\n");

    fp_mul(t0, X1, X2); // t0
    //fp_print("t0 = ",  t0);

    fp_add(t3, X1, Y1); // t3
    //fp_print("t3 = ",  t3);

    fp_add(X1, X1, Z1); // td
    //fp_print("td = ",  X1);

    fp_add(t2, X2, Z2); // te
    //fp_print("te = ",  t2);

    fp_mul(X1, X1, t2); // tf
    //fp_print("tf = ",  X1);


    fp_add(t1, Y2, Z2); // t9
    //fp_print("t9 = ",  t1);

    fp_mul(t2, Z1, Z2); // t2
    //fp_print("t2 = ",  t2);

    fp_add(Z1, Z1, Y1); // t8
    //fp_print("t8 = ",  Z1);

    fp_sub(Z2, Z2, Y2); // T9
    //fp_print("T9 = ",  Z2);
    fp_mul(Z2, Z2, Z1); // Ta
    //fp_print("Ta = ",  Z2);
    fp_mul(Z1, Z1, t1); // ta

    //fp_print("ta = ",  Z1);
    fp_sub(Z1, Z1, t2); // tc

    fp_add(t1, X2, Y2); // t4
    //fp_print("t4 = ",  t1);

    fp_mul(t1, t1, t3); // t5
    //fp_print("t5 = ",  t1);

    fp_sub(X1, X1, t2); // (th)
    fp_sub(X2, X2, Y2); // T4
    //fp_print("T4 = ",  X2);
    fp_mul(X2, X2, t3); // T5
    //fp_print("T5 = ",  X2);
    fp_sub(X2, X2, t0); // T7

    fp_mul(Y1, Y1, Y2); // t1
    //fp_print("t1 = ",  Y1);

    fp_x12(Y2, t2);     // tk
    //fp_print("tk = ",  Y2);

    fp_add(Z2, Z2, Y1); // Tc
    fp_sub(Z2, Z2, t2); // Tc
    //fp_print("Tc = ",  Z2);

    fp_sub(t2, t1, t0); // (t7)
    fp_sub(t2, t2, Y1); // t7
    //fp_print("t7 = ",  t2);

    fp_add(X2, X2, Y1); // T7
    //fp_print("T7 = ",  X2);

    fp_sub(Z1, Z1, Y1); // tc
    //fp_print("tc = ",  Z1);

    fp_sub(X1, X1, t0); // th
    //fp_print("th = ",  X1);

    fp_x3(t0, t0);      // ti
    //fp_print("ti = ",  t0);

    fp_x12(X1, X1);     // tn
    //fp_print("tn = ",  X1);


    fp_add(t3, Y2, Y1); // tl
    //fp_print("tl = ",  t3);

    fp_sub(Y1, Y1, Y2); // tm
    //fp_print("tm = ",  Y1);


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

    //fp_print("X3 = ",  X1);
    //fp_print("Y3 = ",  Y1);
    //fp_print("Z3 = ",  Z1);

    //fp_print("X4 = ",  X2);
    //fp_print("Y4 = ",  Y2);
    //fp_print("Z4 = ",  Z2);
#endif
}

// vim: ts=4 et sw=4 si
