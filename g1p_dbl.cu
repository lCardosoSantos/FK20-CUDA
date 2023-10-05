// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "g1.cuh"
#include "fp.cuh"
#include "fp_cpy.cuh"
#include "fp_x2.cuh"
#include "fp_x3.cuh"
#include "fp_x4.cuh"
#include "fp_x8.cuh"
#include "fp_x12.cuh"
#include "fp_add.cuh"
#include "fp_sub.cuh"
#include "fp_sqr.cuh"
#include "fp_mul.cuh"
#include "fp_reduce12.cuh"

// Accumulator
#define A a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab
#define AL a0, a1, a2, a3, a4, a5

// Operands
#define B b0, b1, b2, b3, b4, b5
#define C c0, c1, c2, c3, c4, c5

// Temporaries
#define V v0, v1, v2, v3, v4, v5
#define W w0, w1, w2, w3, w4, w5
#define X x0, x1, x2, x3, x4, x5
#define Y y0, y1, y2, y3, y4, y5
#define Z z0, z1, z2, z3, z4, z5

#define T t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb
#define TL t0, t1, t2, t3, t4, t5

/**
 * @brief G1 point doubling. p ← 2*p
 * 
 * @param[in, out] p 
 * @return void 
 */
__device__ void g1p_dbl(g1p_t &p) {

#if 0 //ndef NDEBUG
    if (!g1p_isPoint(p)) {
        g1p_print("ERROR in g1p_dbl(): Invalid point ", p);

        // return invalid point as result
        fp_zero(p.x);
        fp_zero(p.y);
        fp_zero(p.z);

        return;
    }
#endif

#if 1
    g1p_multi(0, &p, NULL, &p, NULL);
    /*
    uint64_t T, V, W, X, Y, Z;

    fp_cpy(X, p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5]);
    fp_cpy(Y, p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5]);
    fp_cpy(Z, p.z[0], p.z[1], p.z[2], p.z[3], p.z[4], p.z[5]);

    fp_mul(T, X, Y);    fp_reduce12(T); fp_cpy(X, TL);
    fp_sqr(T, Z);       fp_reduce12(T); fp_cpy(V, TL);
    fp_x12(V, V);
    fp_mul(T, Z, Y);    fp_reduce12(T); fp_cpy(Z, TL);
    fp_sqr(T, Y);       fp_reduce12(T); fp_cpy(Y, TL);
    fp_x3(W, V);
    fp_sub(W, Y, W);
    fp_mul(T, X, W);    fp_reduce12(T); fp_cpy(X, TL);
    fp_add(Y, Y, V);
    fp_mul(T, W, Y);    fp_reduce12(T); fp_cpy(W, TL);
    fp_sub(Y, Y, V);
    fp_x8(Y, Y);
    fp_x2(X, X);
    fp_mul(T, Z, Y);    fp_reduce12(T); fp_cpy(Z, TL);
    fp_mul(T, Y, V);    fp_reduce12(T); fp_cpy(Y, TL);
    fp_add(Y, Y, W);

    fp_cpy(p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], X);
    fp_cpy(p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5], Y);
    fp_cpy(p.z[0], p.z[1], p.z[2], p.z[3], p.z[4], p.z[5], Z);
    */
#else
    uint64_t A, B, C, V, W, X, Y, Z;
    unsigned call, ret;
    enum {
        L_begin,

        // Return addresses
        L_mul0,
        L_sqr0,
        L_x12,
        L_mul1,
        L_sqr1,
        L_x3,
        L_sub0,
        L_mul2,
        L_x2,
        L_add0,
        L_mul3,
        L_sub1,
        L_x8,
        L_mul4,
        L_mul5,
        L_add1,

        // Functions
        F_x2,
        F_x3,
        F_x8,
        F_x12,
        F_add,
        F_sub,
        F_sqr,
        F_mul,
        F_red,

        L_end
    };

    fp_cpy(X, p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5]);
    fp_cpy(Y, p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5]);
    fp_cpy(Z, p.z[0], p.z[1], p.z[2], p.z[3], p.z[4], p.z[5]);

    // Fully inlined code is too big; emulate function calls to compress it.
    // Workaround for compiler bug: use for and switch instead of labels and goto.

    call = L_begin;

    while (call != L_end) switch(call) {

        case F_x2:  fp_x2(AL, AL);      call = ret; break;
        case F_x3:  fp_x3(AL, AL);      call = ret; break;
        case F_x8:  fp_x8(AL, AL);      call = ret; break;
        case F_x12: fp_x12(AL, AL);     call = ret; break;
        case F_add: fp_add(AL, B, C);   call = ret; break;
        case F_sub: fp_sub(AL, B, C);   call = ret; break;
        case F_sqr: fp_sqr(A, B);       call = F_red; break;
        case F_mul: fp_mul(A, B, C);    // fall through to reduction
        case F_red: fp_reduce12(A);     call = ret; break;

        case L_begin:
        fp_cpy(B, X);
        fp_cpy(C, Y);
        //fp_mul(A, B, C);
        call = F_mul; ret  = L_mul0;
        break;

        case L_mul0:
        fp_cpy(X, AL);

        fp_cpy(B, Z);
        //fp_sqr(A, B);
        call = F_sqr; ret = L_sqr0;
        break;

        case L_sqr0:
        //fp_x12(AL, AL);
        call = F_x12; ret = L_x12;
        break;

        case L_x12:
        fp_cpy(V, AL);

        fp_cpy(B, Z);
        fp_cpy(C, Y);
        //fp_mul(A, B, C);
        call = F_mul; ret = L_mul1;
        break;

        case L_mul1:
        fp_cpy(Z, AL);

        fp_cpy(B, Y);
        //fp_sqr(A, B);
        call = F_sqr; ret = L_sqr1;
        break;

        case L_sqr1:
        fp_cpy(Y, AL);

        fp_cpy(AL, V);
        //fp_x3(AL, AL);
        call = F_x3; ret = L_x3;
        break;

        case L_x3:
        fp_cpy(C, AL);
        fp_cpy(B, Y);
        // fp_sub(AL, B, C);
        call = F_sub; ret = L_sub0;
        break;

        case L_sub0:
        fp_cpy(W, AL);

        fp_cpy(C, AL);
        fp_cpy(B, X);
        //fp_mul(A, B, C);
        call = F_mul; ret = L_mul2;
        break;

        case L_mul2:
        //fp_x2(AL, AL);
        call = F_x2; ret = L_x2;
        break;

        case L_x2:
        fp_cpy(X, AL);

        fp_cpy(B, Y);
        fp_cpy(C, V);
        //fp_add(Y, Y, V);
        call = F_add; ret = L_add0;
        break;

        case L_add0:
        fp_cpy(Y, AL);

        fp_cpy(C, W);
        fp_cpy(B, Y);
        //fp_mul(W, Y, W);
        call = F_mul; ret = L_mul3;
        break;

        case L_mul3:
        fp_cpy(W, AL);

        //fp_cpy(B, Y);
        fp_cpy(C, V);
        //fp_sub(Y, Y, V);
        call = F_sub; ret = L_sub1;
        break;

        case L_sub1:

        //fp_x8(Y, Y);
        call = F_x8; ret = L_x8;
        break;

        case L_x8:
        //fp_cpy(Y, AL);

        fp_cpy(B, Z);
        fp_cpy(C, AL);
        //fp_mul(Z, Z, Y);
        call = F_mul; ret = L_mul4;
        break;

        case L_mul4:
        fp_cpy(Z, AL);

        fp_cpy(B, V);
        //fp_cpy(C, Y);
        //fp_mul(Y, V, Y);
        call = F_mul; ret = L_mul5;
        break;

        case L_mul5:
        //fp_cpy(Y, AL);

        fp_cpy(B, AL);
        fp_cpy(C, W);
        //fp_add(Y, Y, W);
        call = F_add; ret = L_add1;
        break;

        case L_add1:
        fp_cpy(Y, AL);

        call = L_end;
        break;

        default:
        break;
    }

    fp_cpy(p.x[0], p.x[1], p.x[2], p.x[3], p.x[4], p.x[5], X);
    fp_cpy(p.y[0], p.y[1], p.y[2], p.y[3], p.y[4], p.y[5], Y);
    fp_cpy(p.z[0], p.z[1], p.z[2], p.z[3], p.z[4], p.z[5], Z);
#endif
}

/**
 * @brief G1 point doubling. p ← 2*p
 * previous implementation, without unrolling of the PTX to in-register.
 * 
 * @param[in, out] p 
 * @return void 
 */
__device__ void g1p_dbl_noPTX(g1p_t &p) {

    if (!g1p_isPoint(p)) {
        g1p_print("ERROR in g1p_dbl(): Invalid point ", p);

        // return invalid point as result
        fp_zero(p.x);
        fp_zero(p.y);
        fp_zero(p.z);

        return;
    } 

    fp_t x, y, z, v, w;

    fp_cpy(x, p.x);
    fp_cpy(y, p.y);
    fp_cpy(z, p.z);

    fp_mul(x, x, y);
    fp_sqr(v, z);
    fp_x12(v, v);
    fp_mul(z, z, y);
    fp_sqr(y, y);
    fp_x3(w, v);
    fp_sub(w, y, w);
    fp_mul(x, x, w);
    fp_add(y, y, v);
    fp_mul(w, w, y);
    fp_sub(y, y, v);
    fp_x8(y, y);
    fp_x2(x, x);
    fp_mul(z, z, y);
    fp_mul(y, y, v);
    fp_add(y, y, w);

    fp_cpy(p.x, x);
    fp_cpy(p.y, y);
    fp_cpy(p.z, z);

}

// vim: ts=4 et sw=4 si
