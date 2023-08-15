// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "frtest.cuh"

#define ITER 80

/**
 * @brief Test multiply by constant on Fr: 
 * 
 * 2(4x) == 8x
 * 2(2(2(2(2(2x)))) == 4(4(4x)) == 8(8x)
 * 12x+8x == 4(3x+2x)
 * 3(3(3(2(4(8x))))) == 12(12(12x))
 * 
 * @param testval 
 * @return __global__ 
 */
__global__ void FrTestMulConst(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    // 2*4 == 8

    for (int i=0; pass && i<TESTVALS; i++) {
        fr_t x2x4, x8;

        fr_cpy(x2x4, testval[i]);
        fr_cpy(x8,   testval[i]);

        for (int j=0; pass && j<ITER; j++) {
            fr_t x1;
            fr_cpy(x1, x2x4);

            fr_x2(x2x4);
            fr_x4(x2x4);

            fr_x8(x8);

            if (fr_neq(x2x4, x8)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                fr_print("1   : ",  x1);
                fr_print("2*4 : ",  x2x4);
                fr_print("8   : ",  x8);
            }
            ++count;
        }
    }

    // 2*2*2*2*2*2 == 4*4*4 == 8*8

    for (int i=0; pass && i<TESTVALS; i++) {
        fr_t x2, x4, x8;

        fr_cpy(x2, testval[i]);
        fr_cpy(x4, testval[i]);
        fr_cpy(x8, testval[i]);

        for (int j=0; pass && j<ITER; j++) {
            fr_t x1;
            fr_cpy(x1, x2);

            fr_x2(x2);
            fr_x2(x2);
            fr_x2(x2);
            fr_x2(x2);
            fr_x2(x2);
            fr_x2(x2);

            fr_x4(x4);
            fr_x4(x4);
            fr_x4(x4);

            fr_x8(x8);
            fr_x8(x8);

            if (fr_neq(x2, x4) || fr_neq(x2, x8)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                fr_print("1   : ",  x1);
                fr_print("2^6 : ",  x2);
                fr_print("4^3 : ",  x4);
                fr_print("8^2 : ",  x8);
            }
            ++count;
        }
    }

    // 3*4 == 12

    for (int i=0; pass && i<TESTVALS; i++) {
        fr_t x3x4, x12;

        fr_cpy(x3x4, testval[i]);
        fr_cpy(x12,  testval[i]);

        for (int j=0; pass && j<ITER; j++) {
            fr_t x1;
            fr_cpy(x1, x3x4);

            fr_x3(x3x4);
            fr_x4(x3x4);

            fr_x12(x12);

            if (fr_neq(x3x4, x12)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                fr_print("1   : ",  x1);
                fr_print("3*4 : ",  x3x4);
                fr_print("12  : ",  x12);
            }
            ++count;
        }
    }

    // 12+8 == 4(3+2)

    for (int i=0; pass && i<TESTVALS; i++) {
        fr_t x1, x2, x3, x8, x12, l, r;

        fr_cpy(l, testval[i]);
        fr_cpy(r, testval[i]);

        for (int j=0; pass && j<ITER; j++) {

            fr_cpy(x1, l);

            fr_cpy(x2, l);
            fr_cpy(x3, l);
            fr_cpy(x8, l);
            fr_cpy(x12, l);

            fr_x2(x2);
            fr_x3(x3);
            fr_x8(x8);
            fr_x12(x12);

            fr_cpy(l, x12);
            fr_add(l, x8);

            fr_cpy(r, x3);
            fr_add(r, x2);
            fr_x4(r);

            if (fr_neq(l, r)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", i);
                fr_print("1      : ",  x1);
                fr_print("12+8   : ",  l);
                fr_print("4(3+2) : ",  r);
            }
            ++count;
        }
    }

    // 3*3*3*2*4*8 == 12*12*12

    for (int i=0; pass && i<TESTVALS; i++) {
        fr_t x1, l, r;

        fr_cpy(l, testval[i]);
        fr_cpy(r, testval[i]);

        for (int j=0; pass && j<ITER; j++) {

            fr_cpy(x1, l);

            fr_x3(l);
            fr_x3(l);
            fr_x3(l);
            fr_x2(l);
            fr_x4(l);
            fr_x8(l);

            fr_x12(r);
            fr_x12(r);
            fr_x12(r);

            if (fr_neq(l, r)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", i);
                fr_print("1           : ",  x1);
                fr_print("3*3*3*2*4*8 : ",  l);
                fr_print("12*12*12    : ",  r);
            }
            ++count;
        }
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
