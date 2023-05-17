// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

__global__ void FpTestAdd(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    uint64_t t[6] = {   2, 0, 0, 0, 0, 0 };
    uint64_t u[6] = { tid, 0, 0, 0, 0, 0 };

    fp_t x, y, z;

    fp_fromUint64(x, t);    // x = 2
    fp_fromUint64(y, u);    // y = tid

    if (x[0] != 2) pass = false;
    if (x[1] != 0) pass = false;
    if (x[2] != 0) pass = false;
    if (x[3] != 0) pass = false;
    if (x[4] != 0) pass = false;
    if (x[5] != 0) pass = false;

    if (y[0] != tid) pass = false;
    if (y[1] !=   0) pass = false;
    if (y[2] !=   0) pass = false;
    if (y[3] !=   0) pass = false;
    if (y[4] !=   0) pass = false;
    if (y[5] !=   0) pass = false;

    if (!pass) {
        printf("%d: FAILED after fp_fromUint64\n", tid);
        goto done;
    }

    fp_add(z, y, x);    // z = y + x

    fp_reduce6(z);

    if (z[0] != tid+2) pass = false;
    if (z[1] !=     0) pass = false;
    if (z[2] !=     0) pass = false;
    if (z[3] !=     0) pass = false;
    if (z[4] !=     0) pass = false;
    if (z[5] !=     0) pass = false;

    if (!pass) {
        printf("%d: FAILED after fp_add\n", tid);
        goto done;
    }

done:
    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
