// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"
#include "frtest.cuh"

// x == y-(y-x)

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
                printf("x = "); fr_print(testval[i]);
                printf("y = "); fr_print(testval[j]);
                printf("y-x = "); fr_print(a);
                printf("y-(y-x) = "); fr_print(b);
            }
            ++count;
        }
    }

    // x
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
