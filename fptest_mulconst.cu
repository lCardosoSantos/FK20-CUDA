// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fptest.cuh"
#include "fp_ptx.cuh"

#define ITER 100

/**
 * @brief Test self consistency in multiplication by constant:
 * 
 * 2(4x) = =8x
 * 2(2(2(2(2(2x))))) == 4(4(4x)) == 8(8x)
 * 3(4x) == 12(x)
 * 3(3(3(2(4(8x))))) == 12(12(12x))
 * 
 * @param testval 
 * @return void 
 */
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
                fp_print("1   : ",  x1);
                fp_print("2*4 : ",  x2x4);
                fp_print("8   : ",  x8);
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
                fp_print("1   : ",  x1);
                fp_print("2^6 : ",  x2);
                fp_print("4^3 : ",  x4);
                fp_print("8^2 : ",  x8);
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
                fp_print("1   : ",  x1);
                fp_print("3*4 : ",  x3x4);
                fp_print("12  : ",  x12);
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
                fp_print("1      : ",  x1);
                fp_print("12+8   : ",  l);
                fp_print("4(3+2) : ",  r);
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
                fp_print("1           : ",  x1);
                fp_print("3*3*3*2*4*8 : ",  l);
                fp_print("12*12*12    : ",  r);
            }
            ++count;
        }
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}


/**
 * @brief Test self consistency in multiplication by constant:
 * 
 * 2(4x) = =8x
 * 2(2(2(2(2(2x))))) == 4(4(4x)) == 8(8x)
 * 3(4x) == 12(x)
 * 3(3(3(2(4(8x))))) == 12(12(12x))
 * 
 * @param testval 
 * @return void 
 */
__global__ void FpTestMulConstPTX(testval_t *testval) {

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

            fp_x2_ptx(x2x4, x2x4);
            fp_x4_ptx(x2x4, x2x4);

            fp_x8_ptx(x8, x8);

            if (fp_neq(x2x4, x8)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                fp_print("1   : ",  x1);
                fp_print("2*4 : ",  x2x4);
                fp_print("8   : ",  x8);
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

            fp_x2_ptx(x2, x2);
            fp_x2_ptx(x2, x2);
            fp_x2_ptx(x2, x2);
            fp_x2_ptx(x2, x2);
            fp_x2_ptx(x2, x2);
            fp_x2_ptx(x2, x2);

            fp_x4_ptx(x4, x4);
            fp_x4_ptx(x4, x4);
            fp_x4_ptx(x4, x4);

            fp_x8_ptx(x8, x8);
            fp_x8_ptx(x8, x8);

            if (fp_neq(x2, x4) || fp_neq(x2, x8)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                fp_print("1   : ",  x1);
                fp_print("2^6 : ",  x2);
                fp_print("4^3 : ",  x4);
                fp_print("8^2 : ",  x8);
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

            fp_x3_ptx(x3x4, x3x4);
            fp_x4_ptx(x3x4, x3x4);

            fp_x12_ptx(x12, x12);

            if (fp_neq(x3x4, x12)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                fp_print("1   : ",  x1);
                fp_print("3*4 : ",  x3x4);
                fp_print("12  : ",  x12);
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

            fp_x2_ptx(x2, l);
            fp_x3_ptx(x3, l);
            fp_x8_ptx(x8, l);
            fp_x12_ptx(x12, l);

            fp_add(l, x12, x8);

            fp_add(r, x3, x2);
            fp_x4_ptx(r, r);

            if (fp_neq(l, r)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", i);
                fp_print("1      : ",  x1);
                fp_print("12+8   : ",  l);
                fp_print("4(3+2) : ",  r);
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

            fp_x3_ptx(l, l);
            fp_x3_ptx(l, l);
            fp_x3_ptx(l, l);
            fp_x2_ptx(l, l);
            fp_x4_ptx(l, l);
            fp_x8_ptx(l, l);

            fp_x12_ptx(r, r);
            fp_x12_ptx(r, r);
            fp_x12_ptx(r, r);

            if (fp_neq(l, r)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", i);
                fp_print("1           : ",  x1);
                fp_print("3*3*3*2*4*8 : ",  l);
                fp_print("12*12*12    : ",  r);
            }
            ++count;
        }
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
