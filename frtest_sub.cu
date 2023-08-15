// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "frtest.cuh"

/**
 * @brief Test of subtraction
 * x == y-(y-x)

 * @param testval 
 * @return __global__ 
 */
__global__ void FrTestSub(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    // x == y-(y-x)

    for (int i=0; i<TESTVALS; i++) {
        for (int j=0; j<TESTVALS; j++) {
            fr_t a, b;

            fr_cpy(a, testval[j]);  // y
            fr_cpy(b, testval[j]);  // y

            fr_sub(a, testval[i]);  // y - x
            fr_sub(b, a);           // y - (y - x)

            if (fr_neq(b, testval[i])) {
                pass = false;

                printf("%d,%d: FAILED: inconsistent result\n", i, j);
                fr_print("x = ",  testval[i]);
                fr_print("y = ",  testval[j]);
                fr_print("y-x = ",  a);
                fr_print("y-(y-x) = ",  b);
            }
            ++count;
        }
    }

    // x
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
