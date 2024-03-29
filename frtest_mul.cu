// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "frtest.cuh"

/**
 * @brief Test of the commutative property of multiplication
 * x*y == y*x
 * 
 * @param testval 
 * @return void 
 */
__global__ void FrTestCommutativeMul(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=0; i<TESTVALS; i++) {
        for (int j=0; j<TESTVALS; j++) {
            fr_t x, y;

            fr_cpy(x, testval[i]);
            fr_cpy(y, testval[j]);

            fr_mul(x, testval[j]);  // x * y
            fr_mul(y, testval[i]);  // y * x

            if (fr_neq(x, y)) {
                pass = false;

                printf("%d,%d: FAILED: inconsistent result\n", i, j);
                fr_print("x = ",  testval[i]);
                fr_print("y = ",  testval[j]);
                fr_print("x*y = ",  x);
                fr_print("y*x = ",  y);
            }
            ++count;
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

/**
 * @brief Test of the associative property of multiplication
 * (x*y)*z == x*(y*z)
 * 
 * @param testval 
 * @return void 
 */
__global__ void FrTestAssociativeMul(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=0; i<TESTVALS; i++) {
        for (int j=0; j<TESTVALS; j++) {
            for (int k=0; k<TESTVALS; k++) {
                fr_t a, b, c;

                fr_cpy(a, testval[i]);  // x
                fr_cpy(b, testval[j]);  // y
                fr_cpy(c, testval[i]);  // x

                fr_mul(a, testval[j]);  // x * y
                fr_mul(a, testval[k]);  // (x * y) * z

                fr_mul(b, testval[k]);  // y * z
                fr_mul(c, b);           // x * (y * z)

                if (fr_neq(a, c)) {
                    pass = false;

                    printf("%d,%d,%d: FAILED: inconsistent result\n", i, j, k);
                    fr_print("x = ",  testval[i]);
                    fr_print("y = ",  testval[j]);
                    fr_print("z = ",  testval[k]);
                    fr_print("(x*y)*z = ",  a);
                    fr_print("x*(y*z) = ",  c);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
