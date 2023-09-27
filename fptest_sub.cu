// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fptest.cuh"
#include "fp_ptx.cuh"


/**
 * @brief Test for subtraction in Fp.
 * 
 * 2x == 3x - x
 * 
 * @param testval 
 * @return __global__ 
 */
__global__ void FpTestSub(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fp_t x, l, r;

    // 2x == 3x - x

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_cpy(x, testval[i]);

        fp_x2(l, x);

        fp_x3(r, x);
        fp_sub(r, r, x);

        if (fp_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            fp_print("x    : ",  x);
            fp_print("2x   : ",  l);
            fp_print("3x-x : ",  r);
        }
        ++count;
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

/**
 * @brief Test for subtraction in Fp using PTX macros.
 * 
 * 2x == 3x - x
 * 
 * @param testval 
 * @return __global__ 
 */
__global__ void FpTestSubPTX(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fp_t x, l, r;

    // 2x == 3x - x

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_cpy(x, testval[i]);

        fp_x2(l, x);

        fp_x3(r, x);
        fp_sub_ptx(r, r, x);

        if (fp_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            fp_print("x    : ",  x);
            fp_print("2x   : ",  l);
            fp_print("3x-x : ",  r);
        }
        ++count;
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}


// vim: ts=4 et sw=4 si
