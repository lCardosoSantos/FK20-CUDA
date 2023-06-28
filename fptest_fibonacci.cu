// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

#define ITERATIONS 100000

__global__ void FpTestFibonacci(testval_t *) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fp_t x, y, t, u;

    fp_one(x);
    fp_one(y);

    for (int i=0; i<ITERATIONS; i++) {

        fp_cpy(t, x);
        fp_add(x, x, y);

        fp_cpy(u, x);
        fp_sub(u, u, t);

        if (fp_neq(u, y)) {
            printf("x ="); fp_print(x);
            printf("y ="); fp_print(y);
            printf("x+y ="); fp_print(t);
            printf("x+y-x ="); fp_print(u);
            pass = false;
            break;
        }

        ++count;

        fp_cpy(t, y);
        fp_add(y, y, x);

        fp_cpy(u, y);
        fp_sub(u, u, t);

        if (fp_neq(u, x)) {
            printf("x ="); fp_print(x);
            printf("y ="); fp_print(y);
            printf("x+y ="); fp_print(t);
            printf("x+y-y ="); fp_print(u);
            pass = false;
            break;
        }

        ++count;
    }

    for (int i=0; i<ITERATIONS; i++) {
        fp_sub(y, y, x);
        fp_sub(x, x, y);
    }

    if (!fp_isone(x) || !fp_isone(y)) {
        printf("Reverse iteration failed\n");
        printf("x ="); fp_print(x);
        printf("y ="); fp_print(y);
        pass = false;
    }
    else
        ++count;

    printf("%ld tests passed\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
