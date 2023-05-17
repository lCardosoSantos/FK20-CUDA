// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

__global__ void FpTestMul(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    if (tid > 0xffffffff)
        return;

    uint64_t t[6] = { tid, 0, 0, 0, 0, 0 };

    fp_t x, z;

    fp_fromUint64(x, t);    // x = tid

    if (x[0] != tid) pass = false;
    if (x[1] !=   0) pass = false;
    if (x[2] !=   0) pass = false;
    if (x[3] !=   0) pass = false;
    if (x[4] !=   0) pass = false;
    if (x[5] !=   0) pass = false;

    if (!pass) {
        printf("%d: FAILED after fp_fromUint64\n", tid);
        goto done;
    }

    fp_mul(z, x, x);    // z = tid * tid

    if (z[0] != tid*tid) pass = false;
    if (z[1] !=       0) pass = false;
    if (z[2] !=       0) pass = false;
    if (z[3] !=       0) pass = false;
    if (z[4] !=       0) pass = false;
    if (z[5] !=       0) pass = false;

    if (!pass) {
        printf("%d: FAILED after fp_mul\n", tid);
        goto done;
    }

done:
    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
