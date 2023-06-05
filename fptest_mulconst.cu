// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

#define ITER 80

__global__ void FpTestMulConst(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    // 2*4 == 8

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_t x2x4, x8;

        fp_cpy(x2x4, testval[i]);
        fp_cpy(x8,   testval[i]);

        for (int j=0; pass && j<ITER; j++) {
            fp_t x1;
            fp_cpy(x1, x2x4);

            fp_x2(x2x4, x2x4);
            fp_x4(x2x4, x2x4);

            fp_x8(x8, x8);

            if (fp_neq(x2x4, x8)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                printf("1   : "); fp_print(x1);
                printf("2*4 : "); fp_print(x2x4);
                printf("8   : "); fp_print(x8);
            }
            ++count;
        }
    }

    // 2*2*2*2*2*2 == 4*4*4 == 8*8

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_t x2, x4, x8;

        fp_cpy(x2, testval[i]);
        fp_cpy(x4, testval[i]);
        fp_cpy(x8, testval[i]);

        for (int j=0; pass && j<ITER; j++) {
            fp_t x1;
            fp_cpy(x1, x2);

            fp_x2(x2, x2);
            fp_x2(x2, x2);
            fp_x2(x2, x2);
            fp_x2(x2, x2);
            fp_x2(x2, x2);
            fp_x2(x2, x2);

            fp_x4(x4, x4);
            fp_x4(x4, x4);
            fp_x4(x4, x4);

            fp_x8(x8, x8);
            fp_x8(x8, x8);

            if (fp_neq(x2, x4) || fp_neq(x2, x8)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                printf("1   : "); fp_print(x1);
                printf("2^6 : "); fp_print(x2);
                printf("4^3 : "); fp_print(x4);
                printf("8^2 : "); fp_print(x8);
            }
            ++count;
        }
    }

    // 3*4 == 12

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_t x3x4, x12;

        fp_cpy(x3x4, testval[i]);
        fp_cpy(x12,  testval[i]);

        for (int j=0; pass && j<ITER; j++) {
            fp_t x1;
            fp_cpy(x1, x3x4);

            fp_x3(x3x4, x3x4);
            fp_x4(x3x4, x3x4);

            fp_x12(x12, x12);

            if (fp_neq(x3x4, x12)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                printf("1   : "); fp_print(x1);
                printf("3*4 : "); fp_print(x3x4);
                printf("12  : "); fp_print(x12);
            }
            ++count;
        }
    }

    // 12+8 == 4(3+2)

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_t x1, x2, x3, x8, x12, l, r;

        fp_cpy(l, testval[i]);
        fp_cpy(r, testval[i]);

        for (int j=0; pass && j<ITER; j++) {

            fp_cpy(x1, l);

            fp_x2(x2, l);
            fp_x3(x3, l);
            fp_x8(x8, l);
            fp_x12(x12, l);

            fp_add(l, x12, x8);

            fp_add(r, x3, x2);
            fp_x4(r, r);

            if (fp_neq(l, r)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", i);
                printf("1      : "); fp_print(x1);
                printf("12+8   : "); fp_print(l);
                printf("4(3+2) : "); fp_print(r);
            }
            ++count;
        }
    }

    // 3*3*3*2*4*8 == 12*12*12

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_t x1, l, r;

        fp_cpy(l, testval[i]);
        fp_cpy(r, testval[i]);

        for (int j=0; pass && j<ITER; j++) {

            fp_cpy(x1, l);

            fp_x3(l, l);
            fp_x3(l, l);
            fp_x3(l, l);
            fp_x2(l, l);
            fp_x4(l, l);
            fp_x8(l, l);

            fp_x12(r, r);
            fp_x12(r, r);
            fp_x12(r, r);

            if (fp_neq(l, r)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", i);
                printf("1           : "); fp_print(x1);
                printf("3*3*3*2*4*8 : "); fp_print(l);
                printf("12*12*12    : "); fp_print(r);
            }
            ++count;
        }
    }

    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
