// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

/**
 * @brief Test for multiplicative inverse mod p in Fp. 
 *
 * Test for self consistency as x == x*inv(x)*x
 * 
 * @param testval 
 * @return void 
 */
__global__ void FpTestInv(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    uint64_t v[6] = { tid + 2, 0, 0, 0, 0, 0 };

    fp_t x, y, z;

    fp_fromUint64(x, v);    // x = tid + 2

    if (x[0] != tid+2) pass = false;
    if (x[1] !=     0) pass = false;
    if (x[2] !=     0) pass = false;
    if (x[3] !=     0) pass = false;
    if (x[4] !=     0) pass = false;
    if (x[5] !=     0) pass = false;

    if (!pass) {
        printf("%d: FAILED after fp_fromUint64\n", tid);
        goto done;
    }

    fp_inv(y, x);           // y = x^-1

    if (y[0] == tid+2) pass = false;

    if (!pass) {
        printf("%d: FAILED after fp_inv\n", tid);
        goto done;
    }

    fp_mul(z, y, x);        // z = y * x

    fp_reduce6(z);

    if (z[0] != 1) pass = false;
    if (z[1] != 0) pass = false;
    if (z[2] != 0) pass = false;
    if (z[3] != 0) pass = false;
    if (z[4] != 0) pass = false;
    if (z[5] != 0) pass = false;

    if (!pass) {
        printf("%d: FAILED test of inverse\n", tid);
        goto done;
    }

    fp_mul(z, z, x);        // z *= x

    fp_reduce6(z);

    if (z[0] != v[0]) pass = false;
    if (z[1] != v[1]) pass = false;
    if (z[2] != v[2]) pass = false;
    if (z[3] != v[3]) pass = false;
    if (z[4] != v[4]) pass = false;
    if (z[5] != v[5]) pass = false;

    if (!pass) {
        printf("%d: FAILED to match input value\n", tid);
        goto done;
    }

done:
    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
