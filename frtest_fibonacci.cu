// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "frtest.cuh"

#define ITERATIONS 100000

/**
 * @brief Test addition and subtraction in Fr using a fibonacci sequence (chain
 * dependency) from 1 to ITERATIONS and back
 * 
 * @return void 
 */
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
            fr_print("x =",  x);
            fr_print("y =",  y);
            fr_print("x+y =",  t);
            fr_print("x+y-x =",  u);
            pass = false;
            break;
        }

        ++count;

        fr_cpy(t, y);
        fr_add(y, x);

        fr_cpy(u, y);
        fr_sub(u, t);

        if (fr_neq(u, x)) {
            fr_print("x =",  x);
            fr_print("y =",  y);
            fr_print("x+y =",  t);
            fr_print("x+y-y =",  u);
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
        fr_print("x =",  x);
        fr_print("y =",  y);
        pass = false;
    }
    else
        ++count;

    printf("%ld tests passed\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
