// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"
#include "frtest.cuh"

#define ITERATIONS 1000000

__global__ void FrTestFibonacci(testval_t *) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fr_t x, y, t, u;

    fr_one(x);
    fr_one(y);

    for (int i=0; i<ITERATIONS; i++) {

        fr_cpy(t, x);
        fr_add(x, y);

        fr_cpy(u, x);
        fr_sub(u, t);

        if (fr_neq(u, y)) {
            printf("x ="); fr_print(x);
            printf("y ="); fr_print(y);
            printf("x+y ="); fr_print(t);
            printf("x+y-x ="); fr_print(u);
            pass = false;
            break;
        }

        ++count;

        fr_cpy(t, y);
        fr_add(y, x);

        fr_cpy(u, y);
        fr_sub(u, t);

        if (fr_neq(u, x)) {
            printf("x ="); fr_print(x);
            printf("y ="); fr_print(y);
            printf("x+y ="); fr_print(t);
            printf("x+y-y ="); fr_print(u);
            pass = false;
            break;
        }

        ++count;
    }

    for (int i=0; i<ITERATIONS; i++) {
        fr_sub(y, x);
        fr_sub(x, y);
    }

    if (!fr_isone(x) || !fr_isone(y)) {
        printf("Reverse iteration failed\n");
        printf("x ="); fr_print(x);
        printf("y ="); fr_print(y);
        pass = false;
    }
    else
        ++count;

    printf("%ld tests passed\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
