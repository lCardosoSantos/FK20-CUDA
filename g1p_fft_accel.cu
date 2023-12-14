// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

#include "g1p_fft_accel.cuh"

// I/O

#define PX p->x[0], p->x[1], p->x[2], p->x[3], p->x[4], p->x[5]
#define PY p->y[0], p->y[1], p->y[2], p->y[3], p->y[4], p->y[5]
#define PZ p->z[0], p->z[1], p->z[2], p->z[3], p->z[4], p->z[5]

#define QX q->x[0], q->x[1], q->x[2], q->x[3], q->x[4], q->x[5]
#define QY q->y[0], q->y[1], q->y[2], q->y[3], q->y[4], q->y[5]
#define QZ q->z[0], q->z[1], q->z[2], q->z[3], q->z[4], q->z[5]

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

#define X0 X00, X01, X02, X03, X04, X05
#define Y0 Y00, Y01, Y02, Y03, Y04, Y05
#define Z0 Z00, Z01, Z02, Z03, Z04, Z05
#define P0 X0, Y0, Z0

#define X1 X10, X11, X12, X13, X14, X15
#define Y1 Y10, Y11, Y12, Y13, Y14, Y15
#define Z1 Z10, Z11, Z12, Z13, Z14, Z15
#define P1 X1, Y1, Z1

#define X2 X20, X21, X22, X23, X24, X25
#define Y2 Y20, Y21, Y22, Y23, Y24, Y25
#define Z2 Z20, Z21, Z22, Z23, Z24, Z25
#define P2 X2, Y2, Z2

#define X3 X30, X31, X32, X33, X34, X35
#define Y3 Y30, Y31, Y32, Y33, Y34, Y35
#define Z3 Z30, Z31, Z32, Z33, Z34, Z35
#define P3 X3, Y3, Z3

// Library with program index and program contents

__managed__ static uint8_t code[0x2A000];
__managed__ static uint32_t library[513];   // entry points

/**
 * @brief G1p accelerator for 512-point FFT/NTT
 *
 * @param[in, out] p First point
 * @param[in, out] q Second point
 * @param[in] w The power of the root of unity
 * @return void
 *
 * The value of w selects a program which performs an
 * addition-subtraction chain and a butterfly (addsub) operation.
 *
 * For w values of 0-255, a Cooley-Tukey butterfly is performed.
 * For w values of 257-512, a Gentleman-Sande butterfly is performed.
 * For w=256, an addition, division by 512, and duplication are performed.
 */
__device__ void g1p_fft_accel(g1p_t *p, g1p_t *q, unsigned w) {

    uint64_t A, B, C, t0, t1, t2, t3, P0, P1, P2, P3;
    uint8_t *ip; // pointer to next instruction in the G1p program
    uint32_t
        call,   // next state / function to call
        ret;    // return state

    assert(p != nullptr);
    assert(q != nullptr);
    assert(w <= 512);

    assert(gridDim.x == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
//  assert(blockDim.x == 256);
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);

    // Labels for the point arithmetic state machine

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

        // End of doubling or addition

        L_end
    };

    // printf("Running program %d @ %d\n", w, library[w]);

    // Load the starting point of the requested G1p program

    ip = code + library[w];

    // Adjust pointers

    p += threadIdx.x;
    q += threadIdx.x;

    // Outer loop

    while (true) {

        // Load instruction

        uint8_t insn = *ip++;

        // printf("%x: %s\n", insn, insn_name[insn]);

        if (insn >= end)
            break;

        switch(insn) {
            case dbl:
                call = D_begin;
                break;

            case add1:
                fp_cpy(X2, X1);
                fp_cpy(Y2, Y1);
                fp_cpy(Z2, Z1);
                call = A_begin;
                break;

            case sub1:
                fp_cpy(X2, X1);
                fp_neg(Y2, Y1);
                fp_cpy(Z2, Z1);
                call = A_begin;
                break;

            case add3:
                fp_cpy(X2, X3);
                fp_cpy(Y2, Y3);
                fp_cpy(Z2, Z3);
                call = A_begin;
                break;

            case sub3:
                fp_cpy(X2, X3);
                fp_neg(Y2, Y3);
                fp_cpy(Z2, Z3);
                call = A_begin;
                break;

            case sw3:
                fp_cpy(X2, X3);
                fp_cpy(Y2, Y3);
                fp_cpy(Z2, Z3);
                fp_cpy(X3, X0);
                fp_cpy(Y3, Y0);
                fp_cpy(Z3, Z0);
                fp_cpy(X0, X2);
                fp_cpy(Y0, Y2);
                fp_cpy(Z0, Z2);
                call = L_end;
                break;

            case st1:
                fp_cpy(X1, X0);
                fp_cpy(Y1, Y0);
                fp_cpy(Z1, Z0);
                call = L_end;
                break;

            case st3:
                fp_cpy(X3, X0);
                fp_cpy(Y3, Y0);
                fp_cpy(Z3, Z0);
                call = L_end;
                break;

            case ldp:
                fp_cpy(X0, PX);
                fp_cpy(Y0, PY);
                fp_cpy(Z0, PZ);
                call = L_end;
                break;

            case stp:
                fp_cpy(PX, X0);
                fp_cpy(PY, Y0);
                fp_cpy(PZ, Z0);
                call = L_end;
                break;

            case ldq:
                fp_cpy(X0, QX);
                fp_cpy(Y0, QY);
                fp_cpy(Z0, QZ);
                call = L_end;
                break;

            case stq:
                fp_cpy(QX, X0);
                fp_cpy(QY, Y0);
                fp_cpy(QZ, Z0);
                call = L_end;
                break;

            default:
                assert(insn < end);
                break;
        }

        // Fully inlined code is too big; emulate function calls to compress the code size.
        // Workaround for compiler bug: use a loop with a switch instead of labels and goto.

        while (call != L_end) switch(call) {

            //// Fp functions ////

    F_x2:   case F_x2:  fp_x2(AL, AL);      call = ret; break;
    F_x3:   case F_x3:  fp_x3(AL, AL);      call = ret; break;
    F_x8:   case F_x8:  fp_x8(AL, AL);      call = ret; break;
    F_x12:  case F_x12: fp_x12(AL, AL);     call = ret; break;
    F_add:  case F_add: fp_add(AL, B, C);   call = ret; break;
    F_sub:  case F_sub: fp_sub(AL, B, C);   call = ret; break;
    F_sqr:  case F_sqr: fp_sqr(A, B);       goto F_red; break;
    F_mul:  case F_mul: fp_mul(A, B, C);    // fall through to reduction
    F_red:  case F_red: fp_reduce12(AL, A); call = ret; break;

            //// G1 doubling ////

            // P0 ← 2 * P0

            case D_begin:
                fp_cpy(B, X0);
                fp_cpy(C, Y0);
                //fp_mul(A, B, C);
                ret  = D_mul0;
                goto F_mul;
                break;

            case D_mul0:
                fp_cpy(X0, AL);

                fp_cpy(B, Z0);
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

                fp_cpy(B, Z0);
                fp_cpy(C, Y0);
                //fp_mul(A, B, C);
                ret = D_mul1;
                goto F_mul;
                break;

            case D_mul1:
                fp_cpy(Z0, AL);

                fp_cpy(B, Y0);
                //fp_sqr(A, B);
                ret = D_sqr1;
                goto F_sqr;
                break;

            case D_sqr1:
                fp_cpy(Y0, AL);

                fp_cpy(AL, t0);
                //fp_x3(AL, AL);
                ret = D_x3;
                goto F_x3;
                break;

            case D_x3:
                fp_cpy(C, AL);
                fp_cpy(B, Y0);
                // fp_sub(AL, B, C);
                ret = D_sub0;
                goto F_sub;
                break;

            case D_sub0:
                fp_cpy(t1, AL);

                fp_cpy(C, AL);
                fp_cpy(B, X0);
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
                fp_cpy(X0, AL);

                fp_cpy(B, Y0);
                fp_cpy(C, t0);
                //fp_add(Y0, Y0, t0);
                ret = D_add0;
                goto F_add;
                break;

            case D_add0:
                fp_cpy(Y0, AL);

                fp_cpy(C, t1);
                fp_cpy(B, Y0);
                //fp_mul(t1, Y0, t1);
                ret = D_mul3;
                goto F_mul;
                break;

            case D_mul3:
                fp_cpy(t1, AL);

                //fp_cpy(B, Y0);
                fp_cpy(C, t0);
                //fp_sub(Y0, Y0, t0);
                ret = D_sub1;
                goto F_sub;
                break;

            case D_sub1:

                //fp_x8(Y0, Y0);
                ret = D_x8;
                goto F_x8;
                break;

            case D_x8:
                //fp_cpy(Y0, AL);

                fp_cpy(B, Z0);
                fp_cpy(C, AL);
                //fp_mul(Z0, Z0, Y0);
                ret = D_mul4;
                goto F_mul;
                break;

            case D_mul4:
                fp_cpy(Z0, AL);

                fp_cpy(B, t0);
                //fp_cpy(C, Y0);
                //fp_mul(Y0, t0, Y0);
                ret = D_mul5;
                goto F_mul;
                break;

            case D_mul5:
                //fp_cpy(Y0, AL);

                fp_cpy(B, AL);
                fp_cpy(C, t1);
                //fp_add(Y0, Y0, t1);
                ret = D_add1;
                goto F_add;
                break;

            case D_add1:
                fp_cpy(Y0, AL);

                call = L_end;
                break;

            //// G1 addition ////

            // P0 ← P0 + P2

            case A_begin:

#if 0
                fp_cpy(B, X0);
                fp_cpy(C, Y0);
                //fp_add(t0, X0, Y0); // t3
                ret = A_add0;
                call = F_add;
                break;

            case A_add0:
                fp_cpy(t0, AL);

                fp_cpy(B, Z0);
                //fp_add(t1, Y0, Z0); // t8
                ret = A_add1;
                call = F_add;
                break;

            case A_add1:
                fp_cpy(t1, AL);

                fp_cpy(C, X0);
                // fp_add(t2, Z0, X0); // td
                ret = A_add2;
                call = F_add;
                break;

            case A_add2:
                fp_cpy(t2, AL);

            ////////
#else
                fp_add(t0, X0, Y0); // t3
                fp_add(t1, Y0, Z0); // t8
                fp_add(t2, Z0, X0); // td
                fp_cpy(C, X0);
#endif
                fp_cpy(B, X2);
                //fp_mul(X0, X0, X2); // t0
                ret = A_mul0;
                goto F_mul;
                break;

            case A_mul0:
                fp_cpy(X0, AL);

                fp_cpy(B, Z0);
                fp_cpy(C, Z2);
                //fp_mul(Z0, Z0, Z2); // t2
                ret = A_mul1;
                goto F_mul;
                break;

            case A_mul1:
                fp_cpy(Z0, AL);

                fp_cpy(B, Y0);
                fp_cpy(C, Y2);
                //fp_mul(Y0, Y0, Y2); // t1
                ret = A_mul2;
                goto F_mul;
                break;

            case A_mul2:
                fp_cpy(Y0, AL);

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
                fp_cpy(AL, X0);
                //fp_x3(t0, X0);      // ti
                ret = A_x3;
                call = F_x3;
                break;

            case A_x3:
                fp_cpy(t0, AL);

                fp_cpy(B, X0);
                fp_cpy(C, Z0);
                //fp_add(t2, Z0, X0); // tg
                ret = A_add6;
                call = F_add;
                break;

            case A_add6:
                fp_cpy(t2, AL);

                fp_cpy(B, Y0);
                //fp_add(t1, Y0, Z0); // tb
                ret = A_add7;
                call = F_add;
                break;

            case A_add7:
                fp_cpy(t1, AL);

                fp_cpy(AL, Z0); // = C
                //fp_x12(t3, Z0);     // tk
                ret = A_x12a;
                call = F_x12;
                break;

            case A_x12a:
                fp_cpy(t3, AL);

            ////////

                fp_cpy(C, X0);
                //fp_add(X0, X0, Y0); // t6
                ret = A_add8;
                call = F_add;
                break;

            case A_add8:
                fp_cpy(X0, AL);

                fp_cpy(C, t3);
                //fp_add(Z0, Y0, t3); // tl
                ret = A_add9;
                call = F_add;
                break;

            case A_add9:
                fp_cpy(Z0, AL);

                //fp_sub(Y0, Y0, t3); // tm
                ret = A_sub0;
                call = F_sub;
                break;

            case A_sub0:
                fp_cpy(Y0, AL);

            ////////

                fp_cpy(B, X2);
                fp_cpy(C, X0);
                //fp_sub(X0, X2, X0); // t7
                ret = A_sub1;
                call = F_sub;
                break;
            case A_sub1:
                fp_cpy(X0, AL);

                fp_cpy(C, X0);
#else
                fp_x3(t0, X0);      // ti
                fp_add(t2, Z0, X0); // tg
                fp_add(t1, Y0, Z0); // tb
                fp_x12(t3, Z0);     // tk
                fp_add(X0, X0, Y0); // t6
                fp_add(Z0, Y0, t3); // tl
                fp_sub(Y0, Y0, t3); // tm
                fp_sub(C, X2, X0);  // t7
#endif

                fp_cpy(B, t0);
                //fp_mul(X2, X0, t0); // ts
                ret = A_mul6;
                goto F_mul;
                break;

            case A_mul6:
                fp_cpy(X2, AL);

            ////////

                fp_cpy(B, Y0);
                //fp_mul(X0, X0, Y0); // tp
                ret = A_mul7;
                goto F_mul;
                break;

            case A_mul7:
                fp_cpy(X0, AL);

                fp_cpy(C, Z0);
                //fp_mul(Y0, Y0, Z0); // tr
                ret = A_mul8;
                goto F_mul;
                break;

            case A_mul8:
                fp_cpy(Y0, AL);

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
                fp_cpy(C, Z0);
                //fp_mul(Z0, Z0, Y2); // tt
                ret = A_mul9;
                goto F_mul;
                break;

            case A_mul9:
                fp_cpy(Z0, AL);

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

                fp_cpy(B, X0);
                fp_cpy(C, Y2);
                //fp_sub(X0, X0, Y2); // X3
                ret = A_sub4;
                call = F_sub;
                break;

            case A_sub4:
                fp_cpy(X0, AL);

                fp_cpy(B, Y0);
                fp_cpy(C, Z2);
                //fp_add(Y0, Y0, Z2); // Y3
                ret = A_add10;
                call = F_add;
                break;

            case A_add10:
                fp_cpy(Y0, AL);

                fp_cpy(B, Z0);
                fp_cpy(C, X2);
                //fp_add(Z0, Z0, X2); // Z3
                ret = A_add11;
                call = F_add;
                break;

            case A_add11:
                fp_cpy(Z0, AL);
#else
                fp_sub(X0, X0, Y2); // X3
                fp_add(Y0, Y0, AL); // Y3
                fp_add(Z0, Z0, X2); // Z3
#endif
                call = L_end;
                break;

            default: call = L_end; break;
        }
    }
}

__global__ void g1p_fft_accel_wrapper(g1p_t *p, g1p_t *q, unsigned w) {
    g1p_fft_accel(p, q, w);

    // Add this to double the number of elements processed by this kernel
#if 0
    g1p_fft_accel(p+blockDim.x, q+blockDim.x, w);
#endif
}

__host__ void g1p_fft_accel_init() {
    uint8_t *ip = code; // instruction pointer
    uint32_t *pi = library; // program index

    for (unsigned w=0; w<513; w++) {

        // Save entry point

        *pi++ = ip - code;

        // 0 and 512 multiply by 1, so for these we only perform the butterfly.

        if ((w % 512) == 0) {
            *ip++ = ldq;
            *ip++ = st1;
            *ip++ = ldp;
            *ip++ = st3;
            *ip++ = add1;
            *ip++ = stp;
            *ip++ = sw3;
            *ip++ = sub1;
            *ip++ = stq;
            *ip++ = end;
            continue;
        }

        std::vector<unsigned> mul;

        int i;
        int state;  // borrow
        unsigned next;  // value of the next several multiplier bits considered
        size_t dbl_count = 0, add_count = 0;

        // Start in state 0 (no borrow)

        state = 0;

        // Prepend 0

        mul.push_back(0);

        // Read multiplier into vector, msb to lsb

        for (int word=3; word>=0; word--)
            for (int bit=63; bit>=0; bit--)
                mul.push_back((fr_roots_host[w][word] >> bit) & 1);

        // Skip leading zeros

        for (i=0; (i < mul.size()) && (mul[i] == 0); i++);

        assert((i+1) < mul.size()); // Not supposed to multiply by 0 or 1.

        // for (auto bit:mul) printf("%x", bit); printf("\n");
        // for (int j=0; j<i; j++) putchar(' '); for (int j=i; j<mul.size(); j++) printf("%x", mul[j]); printf("\n\n");

        // Append zeros

        for (int j=0; j<4; j++)
            mul.push_back(0);

        // Start

        *ip++ = ldq;

        // Inverse FFT: butterfly before multiplication
        if (w > 256) {
            *ip++ = st1;    // P1 = Q
            *ip++ = ldp;    // P0 = P
            *ip++ = st3;    // P3 = P
            *ip++ = add1;   // P0 = P+Q
            *ip++ = stp;
            *ip++ = sw3;    // P0 = P
            *ip++ = sub1;   // P0 = P-Q
        }

        // 256: Special case for FK20; last IFT stage + zero + first FFT stage, no butterfly

        if (w == 256) {
            *ip++ = st1;
            *ip++ = ldp;
            *ip++ = add1;   // P0 = P+Q
        }

        next = 0;
        next += mul[i+0] << 4;
        next += mul[i+1] << 3;
        next += mul[i+2] << 2;
        next += mul[i+3] << 1;
        next += mul[i+4] << 0;

        // printf("%x%x%x%x%x\n", mul[i+0], mul[i+1], mul[i+2], mul[i+3], mul[i+4]);

        /*
            Process the first multiplier bit while ensuring P1 and P3 end up containg P0 and 3*P0, respectively.

            11111* -> 111*  P0 =  8*P0, state = 1:  st1, dbl, st3, add1, sw3, dbl, dbl
            11110* -> 10*   P0 = 16*P0, state = 1:  st1, dbl, st3, add1, sw3, dbl, dbl, dbl
            11101* -> 1*    P0 = 30*P0, state = 1:  st1, dbl, add1, st3, dbl, dbl, add3, dbl
            11100* -> 0*    P0 = 29*P0, state = 0:  st1, dbl, add1, st3, dbl, add1, dbl, dbl
            110*   -> 0*    P0 =  6*P0, state = 0:  st1, dbl, add1, st3, dbl
            101*   -> 1*    P0 =  6*P0, state = 1:  st1, dbl, add1, st3, dbl
            100*   -> 0*    P0 =  4*P0, state = 0:  st1, dbl, st3, add1, sw3, dbl
        */

        switch(next) {
            case 0b11111:
                // printf("11111\n");
                *ip++ = st1;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = st3;
                *ip++ = add1;   ++add_count;
                *ip++ = sw3;
                *ip++ = dbl;    ++dbl_count;
                state = 1; i += 2;
                break;
            case 0b11110:
                // printf("11110\n");
                *ip++ = st1;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = st3;
                *ip++ = add1;   ++add_count;
                *ip++ = sw3;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = dbl;    ++dbl_count;
                state = 1; i += 3;
                break;
            case 0b11101:
                // printf("11101\n");
                *ip++ = st1;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = add1;   ++add_count;
                *ip++ = st3;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = add3;   ++add_count;
                state = 1; i += 4;
                break;
            case 0b11100:
                // printf("11100\n");
                *ip++ = st1;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = add1;   ++add_count;
                *ip++ = st3;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = add1;   ++add_count;
                state = 0; i += 4;
                if (mul.size() > 8) {
                    *ip++ = dbl;    ++dbl_count;
                }
                break;
            case 0b11011:
            case 0b11010:
            case 0b11001:
            case 0b11000:
                // printf("110*\n");
                *ip++ = st1;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = add1;   ++add_count;
                *ip++ = st3;
                state = 0; i += 2;
                break;
            case 0b10111:
            case 0b10110:
            case 0b10101:
            case 0b10100:
                // printf("101*\n");
                *ip++ = st1;
                *ip++ = dbl;    ++dbl_count;
                *ip++ = add1;   ++add_count;
                *ip++ = st3;
                state = 1; i += 2;
                break;
            case 0b10010:
            case 0b10001:
                if (mul.size() <= 10) {
                    // printf("10010/10001\n");
                    state = 0; i += 2;
                    *ip++ = dbl;    ++dbl_count;
                    break;
                }
            case 0b10011:
                // printf("100*\n");
                state = 0; i += 2;
                if (mul.size() > 6) {
                    *ip++ = st1;
                    *ip++ = dbl;    ++dbl_count;
                    *ip++ = st3;
                    *ip++ = add1;   ++add_count;
                    *ip++ = sw3;
                }
                break;
            case 0b10000:
                // printf("10000\n");
                state = 0; i += 2;
                if (mul.size() > 10) {
                    *ip++ = st1;
                    *ip++ = dbl;    ++dbl_count;
                    *ip++ = st3;
                    *ip++ = add1;   ++add_count;
                    *ip++ = sw3;
                }
                if (mul.size() <= 10) {
                    *ip++ = dbl;    ++dbl_count;
                }
                break;
            default:
                fprintf(stderr, "ERROR: invalid value %x at %s:%d\n", next, __FILE__, __LINE__);
                exit(-1);
        }

        while (i+4<mul.size()) {
            next = 0;
            next += mul[i+0] << 3;
            next += mul[i+1] << 2;
            next += mul[i+2] << 1;
            next += mul[i+3] << 0;

            // printf("%2d: %d %c%c%c%c\n", i, state, mul[i+0], mul[i+1], mul[i+2], mul[i+3]);

            if (state == 0)
                switch(next) {
                    case 0b0111:
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = add1;   ++add_count;
                        i++; state = 1; break;
                    case 0b1000:
                    case 0b1001:
                    case 0b1010:
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = add1;   ++add_count;
                        i++; state = 0; break;
                    case 0b1011:
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = add3;   ++add_count;
                        i++;
                        i++; state = 1; break;
                    case 0b1100:
                    case 0b1101:
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = add3;   ++add_count;
                        i++;
                        i++; state = 0; break;
                    case 0b1110:
                    case 0b1111:
                        fprintf(stderr, "ERROR: Reached unreachable state (%d, %d)\n", state, next);
                        exit(-1);

                    default:
                        *ip++ = dbl;    ++dbl_count;
                        i++; break;
                }
            else
                switch(next) {
                    case 0b0000:
                    case 0b0001:
                        fprintf(stderr, "ERROR: Reached unreachable state (%d, %d)\n", state, next);
                        exit(-1);

                    case 0b0010:
                    case 0b0011:
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = sub3;   ++add_count;
                        i++;
                        i++; state = 1; break;
                    case 0b0100:
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = sub3;   ++add_count;
                        i++;
                        i++; state = 0; break;
                    case 0b0101:
                    case 0b0110:
                    case 0b0111:
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = sub1;   ++add_count;
                        i++; state = 1; break;
                    case 0b1000:
                        *ip++ = dbl;    ++dbl_count;
                        *ip++ = sub1;   ++add_count;
                        i++; state = 0; break;

                    default:
                        *ip++ = dbl;    ++dbl_count;
                        i++; break;
                }
        }

        if (w < 256) {

            // wQ is in P0

            *ip++ = st1;    // P1 = wQ
            *ip++ = ldp;
            *ip++ = st3;    // P3 = P
            *ip++ = add1;   // P0 = P+wQ
            *ip++ = stp;
            *ip++ = sw3;    // P0 = P
            *ip++ = sub1;   // P0 = P-wQ
        }

        // 256: P = Q = (P+Q)/512

        if (w == 256) {
            *ip++ = stp;
        }

        *ip++ = stq;
        *ip++ = end;
    }

    assert((ip-code) < sizeof(code));

#ifndef NDEBUG
    printf("%s: %lu programs, %lu instructions\n", __func__, pi-library, ip-code);
#endif
}

// Enable this to generate a standalone test program.

#if 1
__global__ void g1p_fft_accel_test() {
    g1p_t p, q, pref, qref, ptest, qtest;

    g1p_gen(p);
    g1p_gen(q);
    g1p_dbl(q);
    g1p_add(p, q);

    for (unsigned i=0; i<513; i++) {

        g1p_cpy(ptest, p);
        g1p_cpy(qtest, q);

        g1p_fft_accel(&ptest, &qtest, i);

        g1p_cpy(pref, p);
        g1p_cpy(qref, q);

        if (i == 256) {
            g1p_add(qref, pref);
            g1p_mul(qref, fr_roots[513]);
            g1p_cpy(pref, qref);
        }
        else {
            if (i < 256)
                g1p_mul(qref, fr_roots[i]);

            g1p_addsub(pref, qref);

            if (i > 256)
                g1p_mul(qref, fr_roots[i]);
        }

        if (g1p_neq(pref, ptest) || g1p_neq(qref, qtest)) {
            printf("ERROR for i = %d\n", i);
            g1p_print("p", p);
            g1p_print("q", q);
            g1p_print("pout ", ptest);
            g1p_print("qout ", qtest);
            g1p_print("pref ", pref);
            g1p_print("qref ", qref);
            break;
        }

#if 1
        if (g1p_eq(pref, ptest) && g1p_eq(qref, qtest))
            printf("OK %d\n", i);
#endif
    }
}

int main() {

    g1p_fft_accel_init();

    g1p_fft_accel_test<<<1,1>>>();

    cudaDeviceSynchronize();

    return 0;
}
#endif

// vim: ts=4 et sw=4 si foldmethod=syntax
