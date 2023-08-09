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
                fr_print("x = ",  testval[i]);
                fr_print("y = ",  testval[j]);
                fr_print("x+y = ",  a);
                fr_print("x+y = ",  x);
                fr_print("x-y = ",  b);
                fr_print("x-y = ",  y);
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
                    printf("[%d]: ", i); fr_print("", testval[i]);
                    printf("[%d]: ", j); fr_print("", testval[j]);
                    fr_print("a = ",  a);
                    fr_print("x = ",  x);
                    fr_print("b = ",  b);
                    fr_print("y = ",  y);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
