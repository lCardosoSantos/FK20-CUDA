// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"
#include "frtest.cuh"

#define ITER 5

__global__ void FrTestAddSub(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=0; i<TESTVALS; i++) {
        for (int j=0; j<TESTVALS; j++) {
            fr_t a, b, x, y;

            fr_cpy(a, testval[i]);
            fr_cpy(b, testval[i]);

            fr_cpy(x, testval[i]);
            fr_cpy(y, testval[j]);

            fr_addsub(x, y);  // (x,y) = (x+y,x-y)

            fr_add(a, testval[j]);  // x + y
            fr_sub(b, testval[j]);  // x - y

            if (fr_neq(a, x) || fr_neq(b, y)) {
                pass = false;

                printf("%d,%d: FAILED: inconsistent result\n", i, j);
                printf("x = "); fr_print(testval[i]);
                printf("y = "); fr_print(testval[j]);
                printf("x+y = "); fr_print(a);
                printf("x+y = "); fr_print(x);
                printf("x-y = "); fr_print(b);
                printf("x-y = "); fr_print(y);
            }
            ++count;
        }
    }

    for (int i=0; i<TESTVALS; i++) {
        for (int j=0; j<TESTVALS; j++) {
            fr_t a, b, x, y;

            fr_cpy(a, testval[i]);
            fr_cpy(b, testval[j]);

            fr_cpy(x, testval[i]);
            fr_cpy(y, testval[j]);

            for (int k=0; k<ITER; k++) {

                // a,b -> 2a, 2b
                fr_x2(a);
                fr_x2(b);

                // x,y -> 2x, 2y
                fr_addsub(x, y);
                fr_addsub(x, y);

                if (fr_neq(a, x) || fr_neq(b, y)) {
                    pass = false;

                    printf("%d,%d,%d: FAILED: inconsistent result\n", i, j, k);
                    printf("[%d]: ", i); fr_print(testval[i]);
                    printf("[%d]: ", j); fr_print(testval[j]);
                    printf("a = "); fr_print(a);
                    printf("x = "); fr_print(x);
                    printf("b = "); fr_print(b);
                    printf("y = "); fr_print(y);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
