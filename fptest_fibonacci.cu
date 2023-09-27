// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fptest.cuh"
#include "fp_ptx.cuh"

#define ITERATIONS 100000

/**
 * @brief Test addition and subtraction in Fp using a fibonacci sequence (chain
 * dependency) from 1 to ITERATIONS and back
 * 
 * @return void 
 */
__global__ void FpTestFibonacci(testval_t *) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fp_t x, y, t, u, X, Y;

    fp_one(X);
    fp_one(Y);

    for(int ii=0; ii<3; ii++){ //test with different start points for fibonacci
        fp_cpy(x, X);
        fp_cpy(y, Y);
        for (int i=0; i<ITERATIONS; i++) {

            fp_cpy(t, x);
            fp_add(x, x, y);

            fp_cpy(u, x);
            fp_sub(u, u, t);

            if (fp_neq(u, y)) {
                fp_print("x =",  x);
                fp_print("y =",  y);
                fp_print("x+y =",  t);
                fp_print("x+y-x =",  u);
                pass = false;
                break;
            }

            ++count;

            fp_cpy(t, y);
            fp_add(y, y, x);

            fp_cpy(u, y);
            fp_sub(u, u, t);

            if (fp_neq(u, x)) {
                fp_print("x =",  x);
                fp_print("y =",  y);
                fp_print("x+y =",  t);
                fp_print("x+y-y =",  u);
                pass = false;
                break;
            }

            ++count;
        }

        for (int i=0; i<ITERATIONS; i++) {
            fp_sub(y, y, x);
            fp_sub(x, x, y);
        }

        if (fp_neq(x, X) || fp_neq(y, Y)) {
            printf("Reverse iteration failed\n");
            fp_print("x =",  x);
            fp_print("y =",  y);
            pass = false;
        }
        else
            ++count;

        Y[0]+=3;
    }
    printf("%ld tests passed\n", count);

    PRINTPASS(pass);
}

/**
 * @brief Test addition and subtraction in Fp using a fibonacci sequence (chain
 * dependency) from 1 to ITERATIONS and back (Using PTX macros)
 * 
 * @return void 
 */
__global__ void FpTestFibonacciPTX(testval_t *) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fp_t x, y, t, u, X, Y;

    fp_one(X);
    fp_one(Y);

    for(int ii=0; ii<3; ii++){ //test with different start points for fibonacci
        fp_cpy(x, X);
        fp_cpy(y, Y);
        for (int i=0; i<ITERATIONS; i++) {

            fp_cpy(t, x);
            fp_add_ptx(x, x, y);

            fp_cpy(u, x);
            fp_sub_ptx(u, u, t);

            if (fp_neq(u, y)) {
                fp_print("x =",  x);
                fp_print("y =",  y);
                fp_print("x+y =",  t);
                fp_print("x+y-x =",  u);
                pass = false;
                break;
            }

            ++count;

            fp_cpy(t, y);
            fp_add_ptx(y, y, x);

            fp_cpy(u, y);
            fp_sub_ptx(u, u, t);

            if (fp_neq(u, x)) {
                fp_print("x =",  x);
                fp_print("y =",  y);
                fp_print("x+y =",  t);
                fp_print("x+y-y =",  u);
                pass = false;
                break;
            }

            ++count;
        }

        for (int i=0; i<ITERATIONS; i++) {
            fp_sub_ptx(y, y, x);
            fp_sub_ptx(x, x, y);
        }

        if (fp_neq(x, X) || fp_neq(y, Y)) {
            printf("Reverse iteration failed\n");
            fp_print("x =",  x);
            fp_print("y =",  y);
            pass = false;
        }
        else
            ++count;

        Y[0]+=3;
    }
    printf("%ld tests passed\n", count);

    PRINTPASS(pass);
}
// vim: ts=4 et sw=4 si
