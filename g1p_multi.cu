// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#define SUPPORT_DBLADD2 0

#include "g1.cuh"
#include "fp.cuh"
#include "fr.cuh"
#include "fp_cpy.cuh"
#include "fp_neg.cuh"
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

// I/O

#define PX p->x[0], p->x[1], p->x[2], p->x[3], p->x[4], p->x[5]
#define PY p->y[0], p->y[1], p->y[2], p->y[3], p->y[4], p->y[5]
#define PZ p->z[0], p->z[1], p->z[2], p->z[3], p->z[4], p->z[5]

#define QX q->x[0], q->x[1], q->x[2], q->x[3], q->x[4], q->x[5]
#define QY q->y[0], q->y[1], q->y[2], q->y[3], q->y[4], q->y[5]
#define QZ q->z[0], q->z[1], q->z[2], q->z[3], q->z[4], q->z[5]

#define RX r->x[0], r->x[1], r->x[2], r->x[3], r->x[4], r->x[5]
#define RY r->y[0], r->y[1], r->y[2], r->y[3], r->y[4], r->y[5]
#define RZ r->z[0], r->z[1], r->z[2], r->z[3], r->z[4], r->z[5]

#define SX s->x[0], s->x[1], s->x[2], s->x[3], s->x[4], s->x[5]
#define SY s->y[0], s->y[1], s->y[2], s->y[3], s->y[4], s->y[5]
#define SZ s->z[0], s->z[1], s->z[2], s->z[3], s->z[4], s->z[5]

// Accumulator

#define A a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab
#define A7 a0, a1, a2, a3, a4, a5, a6
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
 * @brief G1 point arithmetic toolbox.
 *
 * @param[in] op Operation selector
 * @param[in, out] p
 * @param[in, out] q
 * @param[in] r
 * @param[in] s
 * @return void
 *
 *  -1 Dbl:     p ← 2*r
 *  -2 Add:     p ← r+s
 */
__noinline__
__device__ void g1p_multi(int op, g1p_t *p, g1p_t *q, const g1p_t *r, const g1p_t *s) {

    uint64_t A, B, C, t0, t1, t2, t3, X1, Y1, Z1, X2, Y2, Z2;
    uint32_t
        call,   // next state / function to call
        ret;    // return state

    // Labels for the state machine

    enum {

        // Fp functions

        F_x2,
        F_x3,
        F_x4,
        F_x8,
        F_x12,
        F_add,
        F_sub,
        F_sqr,
        F_mul,
        F_red,

        // G1 doubling

        D_begin,
        D_mul0,
        D_sqr0,
        D_x12,
        D_mul1,
        D_sqr1,
        D_x3,
        D_sub0,
        D_mul2,
        D_x2,
        D_add0,
        D_mul3,
        D_sub1,
        D_x8,
        D_mul4,
        D_mul5,
        D_add1,

        // G1 addition

        A_begin,
        A_add0,
        A_add1,
        A_add2,
        A_mul0,
        A_mul1,
        A_mul2,
        A_add3,
        A_add4,
        A_add5,
        A_mul3,
        A_mul4,
        A_mul5,
        A_x3,
        A_add6,
        A_add7,
        A_x12a,
        A_add8,
        A_add9,
        A_sub0,
        A_sub1,
        A_mul6,
        A_mul7,
        A_mul8,
        A_sub2,
        A_mul9,
        A_sub3,
        A_x12b,
        A_mul10,
        A_mul11,
        A_sub4,
        A_add10,
        A_add11,

        // The end

        L_end
    };

    fp_cpy(X1, RX);
    fp_cpy(Y1, RY);
    fp_cpy(Z1, RZ);

    switch(op) {
        case -1: // Dbl
            call = D_begin;
            break;
        case -2: // Add
            fp_cpy(X2, SX);
            fp_cpy(Y2, SY);
            fp_cpy(Z2, SZ);
            call = A_begin;
            break;
        default:
            printf("%s:%d: Invalid G1 operation %d\n", __FILE__, __LINE__, op);
            call = L_end;
            break;
    }

    // Fully inlined code is too big; emulate function calls to compress it.
    // Workaround for compiler bug: use a loop with a switch instead of labels and goto.

    while (call != L_end) switch(call) {

        //// Fp functions ////

F_x2:   case F_x2:  fp_x2(A7, AL);      goto F_red7;
F_x3:   case F_x3:  fp_x3(A7, AL);      goto F_red7;
F_x8:   case F_x8:  fp_x8(A7, AL);      goto F_red7;
F_x12:  case F_x12: fp_x12(A7, AL);
F_red7:             fp_reduce7(AL, A7); call = ret; break;
F_add:  case F_add: fp_add(AL, B, C);   call = ret; break;
F_sub:  case F_sub: fp_sub(AL, B, C);   call = ret; break;
F_sqr:  case F_sqr: fp_sqr(A, B);       goto F_red; break;
F_mul:  case F_mul: fp_mul(A, B, C);    // fall through to reduction
F_red:  case F_red: fp_reduce12(AL, A); call = ret; break;

        //// G1 doubling ////

        // P0 ← 2 * P0

        case D_begin:
            fp_cpy(B, X1);
            fp_cpy(C, Y1);
            //fp_mul(A, B, C);
            ret  = D_mul0;
            goto F_mul;
            break;

        case D_mul0:
            fp_cpy(X1, AL);

            fp_cpy(B, Z1);
            //fp_sqr(A, B);
            ret = D_sqr0;
            goto F_sqr;
            break;

        case D_sqr0:
            //fp_x12(AL, AL);
            ret = D_x12;
            goto F_x12;
            break;

        case D_x12:
            fp_cpy(t0, AL);

            fp_cpy(B, Z1);
            fp_cpy(C, Y1);
            //fp_mul(A, B, C);
            ret = D_mul1;
            goto F_mul;
            break;

        case D_mul1:
            fp_cpy(Z1, AL);

            fp_cpy(B, Y1);
            //fp_sqr(A, B);
            ret = D_sqr1;
            goto F_sqr;
            break;

        case D_sqr1:
            fp_cpy(Y1, AL);

            fp_cpy(AL, t0);
            //fp_x3(AL, AL);
            ret = D_x3;
            goto F_x3;
            break;

        case D_x3:
            fp_cpy(C, AL);
            fp_cpy(B, Y1);
            // fp_sub(AL, B, C);
            ret = D_sub0;
            goto F_sub;
            break;

        case D_sub0:
            fp_cpy(t1, AL);

            fp_cpy(C, AL);
            fp_cpy(B, X1);
            //fp_mul(A, B, C);
            ret = D_mul2;
            goto F_mul;
            break;

        case D_mul2:
            //fp_x2(AL, AL);
            ret = D_x2;
            goto F_x2;
            break;

        case D_x2:
            fp_cpy(X1, AL);

            fp_cpy(B, Y1);
            fp_cpy(C, t0);
            //fp_add(Y1, Y1, t0);
            ret = D_add0;
            goto F_add;
            break;

        case D_add0:
            fp_cpy(Y1, AL);

            fp_cpy(C, t1);
            fp_cpy(B, Y1);
            //fp_mul(t1, Y1, t1);
            ret = D_mul3;
            goto F_mul;
            break;

        case D_mul3:
            fp_cpy(t1, AL);

            //fp_cpy(B, Y1);
            fp_cpy(C, t0);
            //fp_sub(Y1, Y1, t0);
            ret = D_sub1;
            goto F_sub;
            break;

        case D_sub1:

            //fp_x8(Y1, Y1);
            ret = D_x8;
            goto F_x8;
            break;

        case D_x8:
            //fp_cpy(Y1, AL);

            fp_cpy(B, Z1);
            fp_cpy(C, AL);
            //fp_mul(Z1, Z1, Y1);
            ret = D_mul4;
            goto F_mul;
            break;

        case D_mul4:
            fp_cpy(Z1, AL);

            fp_cpy(B, t0);
            //fp_cpy(C, Y1);
            //fp_mul(Y1, t0, Y1);
            ret = D_mul5;
            goto F_mul;
            break;

        case D_mul5:
            //fp_cpy(Y1, AL);

            fp_cpy(B, AL);
            fp_cpy(C, t1);
            //fp_add(Y1, Y1, t1);
            ret = D_add1;
            goto F_add;
            break;

        case D_add1:
            fp_cpy(Y1, AL);

            call = L_end;
            break;

        //// G1 addition ////

        case A_begin:

            //if (op == -3) printf(" Add\n");
#if 0
            fp_cpy(B, X1);
            fp_cpy(C, Y1);
            //fp_add(t0, X1, Y1); // t3
            ret = A_add0;
            call = F_add;
            break;

        case A_add0:
            fp_cpy(t0, AL);

            fp_cpy(B, Z1);
            //fp_add(t1, Y1, Z1); // t8
            ret = A_add1;
            call = F_add;
            break;

        case A_add1:
            fp_cpy(t1, AL);

            fp_cpy(C, X1);
            // fp_add(t2, Z1, X1); // td
            ret = A_add2;
            call = F_add;
            break;

        case A_add2:
            fp_cpy(t2, AL);

        ////////
#else
            fp_add(t0, X1, Y1); // t3
            fp_add(t1, Y1, Z1); // t8
            fp_add(t2, Z1, X1); // td
            fp_cpy(C, X1);
#endif
            fp_cpy(B, X2);
            //fp_mul(X1, X1, X2); // t0
            ret = A_mul0;
            goto F_mul;
            break;

        case A_mul0:
            fp_cpy(X1, AL);

            fp_cpy(B, Z1);
            fp_cpy(C, Z2);
            //fp_mul(Z1, Z1, Z2); // t2
            ret = A_mul1;
            goto F_mul;
            break;

        case A_mul1:
            fp_cpy(Z1, AL);

            fp_cpy(B, Y1);
            fp_cpy(C, Y2);
            //fp_mul(Y1, Y1, Y2); // t1
            ret = A_mul2;
            goto F_mul;
            break;

        case A_mul2:
            fp_cpy(Y1, AL);

        ////////
#if 0
            fp_cpy(B, X2);
            //fp_add(t3, X2, Y2); // t4
            ret = A_add3;
            call = F_add;
            break;

        case A_add3:
            fp_cpy(t3, AL);

            fp_cpy(B, Z2);
            //fp_add(Y2, Z2, Y2); // t9
            ret = A_add4;
            call = F_add;
            break;

        case A_add4:
            fp_cpy(Y2, AL);

            fp_cpy(C, X2);
            //fp_add(Z2, Z2, X2); // te
            ret = A_add5;
            call = F_add;
            break;

        case A_add5:
            fp_cpy(Z2, AL);
#else
        ////////

            fp_add(t3, X2, Y2); // t4
            fp_add(Y2, Z2, Y2); // t9
            fp_add(Z2, Z2, X2); // te
#endif
            fp_cpy(B, t3);
            fp_cpy(C, t0);
            //fp_mul(X2, t3, t0); // t5
            ret = A_mul3;
            goto F_mul;
            break;

        case A_mul3:
            fp_cpy(X2, AL);

            fp_cpy(B, Y2);
            fp_cpy(C, t1);
            //fp_mul(Y2, Y2, t1); // ta
            ret = A_mul4;
            goto F_mul;
            break;

        case A_mul4:
            fp_cpy(Y2, AL);

            fp_cpy(B, Z2);
            fp_cpy(C, t2);
            //fp_mul(Z2, Z2, t2); // tf
            ret = A_mul5;
            goto F_mul;
            break;

        case A_mul5:
            fp_cpy(Z2, AL);

        ////////
#if 0
            fp_cpy(AL, X1);
            //fp_x3(t0, X1);      // ti
            ret = A_x3;
            call = F_x3;
            break;

        case A_x3:
            fp_cpy(t0, AL);

            fp_cpy(B, X1);
            fp_cpy(C, Z1);
            //fp_add(t2, Z1, X1); // tg
            ret = A_add6;
            call = F_add;
            break;

        case A_add6:
            fp_cpy(t2, AL);

            fp_cpy(B, Y1);
            //fp_add(t1, Y1, Z1); // tb
            ret = A_add7;
            call = F_add;
            break;

        case A_add7:
            fp_cpy(t1, AL);

            fp_cpy(AL, Z1); // = C
            //fp_x12(t3, Z1);     // tk
            ret = A_x12a;
            call = F_x12;
            break;

        case A_x12a:
            fp_cpy(t3, AL);

        ////////

            fp_cpy(C, X1);
            //fp_add(X1, X1, Y1); // t6
            ret = A_add8;
            call = F_add;
            break;

        case A_add8:
            fp_cpy(X1, AL);

            fp_cpy(C, t3);
            //fp_add(Z1, Y1, t3); // tl
            ret = A_add9;
            call = F_add;
            break;

        case A_add9:
            fp_cpy(Z1, AL);

            //fp_sub(Y1, Y1, t3); // tm
            ret = A_sub0;
            call = F_sub;
            break;

        case A_sub0:
            fp_cpy(Y1, AL);

        ////////

            fp_cpy(B, X2);
            fp_cpy(C, X1);
            //fp_sub(X1, X2, X1); // t7
            ret = A_sub1;
            call = F_sub;
            break;
        case A_sub1:
            fp_cpy(X1, AL);

            fp_cpy(C, X1);
#else
            fp_x3(t0, X1);      // ti
            fp_add(t2, Z1, X1); // tg
            fp_add(t1, Y1, Z1); // tb
            fp_x12(t3, Z1);     // tk
            fp_add(X1, X1, Y1); // t6
            fp_add(Z1, Y1, t3); // tl
            fp_sub(Y1, Y1, t3); // tm
            fp_sub(C, X2, X1); // t7
#endif

            fp_cpy(B, t0);
            //fp_mul(X2, X1, t0); // ts
            ret = A_mul6;
            goto F_mul;
            break;

        case A_mul6:
            fp_cpy(X2, AL);

        ////////

            fp_cpy(B, Y1);
            //fp_mul(X1, X1, Y1); // tp
            ret = A_mul7;
            goto F_mul;
            break;

        case A_mul7:
            fp_cpy(X1, AL);

            fp_cpy(C, Z1);
            //fp_mul(Y1, Y1, Z1); // tr
            ret = A_mul8;
            goto F_mul;
            break;

        case A_mul8:
            fp_cpy(Y1, AL);

        ////////

            fp_cpy(B, Y2);
            fp_cpy(C, t1);
            //fp_sub(Y2, Y2, t1); // tc
            ret = A_sub2;
            call = F_sub;
            break;

        case A_sub2:
            fp_cpy(Y2, AL);

            fp_cpy(B, Y2);
            fp_cpy(C, Z1);
            //fp_mul(Z1, Z1, Y2); // tt
            ret = A_mul9;
            goto F_mul;
            break;

        case A_mul9:
            fp_cpy(Z1, AL);

            fp_cpy(B, Z2);
            fp_cpy(C, t2);
            //fp_sub(Z2, Z2, t2); // th
            ret = A_sub3;
            call = F_sub;
            break;

        case A_sub3:
            //fp_cpy(Z2, AL);

        ////////

            //fp_x12(Z2, Z2);     // tn
            ret = A_x12b;
            call = F_x12;
            break;

        case A_x12b:
            fp_cpy(C, AL);

            fp_cpy(B, Y2);
            //fp_mul(Y2, Y2, Z2); // to
            ret = A_mul10;
            goto F_mul;
            break;

        case A_mul10:
            fp_cpy(Y2, AL);

            fp_cpy(B, t0);
            //fp_mul(Z2, Z2, t0); // tq
            ret = A_mul11;
            goto F_mul;
            break;

        case A_mul11:
#if 0
            fp_cpy(Z2, AL);

        ////////

            fp_cpy(B, X1);
            fp_cpy(C, Y2);
            //fp_sub(X1, X1, Y2); // X3
            ret = A_sub4;
            call = F_sub;
            break;

        case A_sub4:
            fp_cpy(X1, AL);

            fp_cpy(B, Y1);
            fp_cpy(C, Z2);
            //fp_add(Y1, Y1, Z2); // Y3
            ret = A_add10;
            call = F_add;
            break;

        case A_add10:
            fp_cpy(Y1, AL);

            fp_cpy(B, Z1);
            fp_cpy(C, X2);
            //fp_add(Z1, Z1, X2); // Z3
            ret = A_add11;
            call = F_add;
            break;

        case A_add11:
            fp_cpy(Z1, AL);
#else
            fp_sub(X1, X1, Y2); // X3
            fp_add(Y1, Y1, AL); // Y3
            fp_add(Z1, Z1, X2); // Z3
#endif
            call = L_end;;
            break;

        default: call = L_end; break;
    }

    fp_cpy(PX, X1);
    fp_cpy(PY, Y1);
    fp_cpy(PZ, Z1);
}

// vim: ts=4 et sw=4 si
