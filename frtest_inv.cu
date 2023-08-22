// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "frtest.cuh"

/**
 * @brief Test for multiplicative inverse mod r in Fr.
 *
 * Test for self consistency as x == x*inv(x)*x
 *
 * @param testval
 * @return void
 */
__global__ void FrTestInv(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    uint64_t v[4] = { tid + 2, 0, 0, 0 };

    fr_t x, y, z;

    fr_fromUint64(x, v);    // x = tid + 2

    if (x[0] != tid+2) pass = false;
    if (x[1] !=     0) pass = false;
    if (x[2] !=     0) pass = false;
    if (x[3] !=     0) pass = false;

    if (!pass) {
        printf("%d: FAILED after fr_fromUint64\n", tid);
        goto done;
    }

    fr_cpy(y, x);
    fr_inv(y);              // y = x^-1

    if (y[0] == tid+2) pass = false;

    if (!pass) {
        printf("%d: FAILED after fr_inv\n", tid);
        goto done;
    }

    fr_cpy(z, y);
    fr_mul(z, x);           // z = y * x

    fr_reduce4(z);

    if (z[0] != 1) pass = false;
    if (z[1] != 0) pass = false;
    if (z[2] != 0) pass = false;
    if (z[3] != 0) pass = false;

    if (!pass) {
        printf("%d: FAILED test of inverse\n", tid);
        goto done;
    }

    fr_mul(z, x);        // z *= x

    fr_reduce4(z);

    if (z[0] != v[0]) pass = false;
    if (z[1] != v[1]) pass = false;
    if (z[2] != v[2]) pass = false;
    if (z[3] != v[3]) pass = false;

    if (!pass) {
        printf("%d: FAILED to match input value\n", tid);
        goto done;
    }

done:
    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
