// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

// x*y == y*x

__global__ void FpTestCommutativeMul(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=0; i<TESTVALS; i++) {
        for (int j=0; j<TESTVALS; j++) {
            fp_t x, y;

            fp_cpy(x, testval[i]);
            fp_cpy(y, testval[j]);

            fp_mul(x, x, testval[j]);  // x * y
            fp_mul(y, y, testval[i]);  // y * x

            if (fp_neq(x, y)) {
                pass = false;

                printf("%d,%d: FAILED: inconsistent result\n", i, j);
                printf("x = "); fp_print(testval[i]);
                printf("y = "); fp_print(testval[j]);
                printf("x*y = "); fp_print(x);
                printf("y*x = "); fp_print(y);
            }
            ++count;
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// (x*y)*z == x*(y*z)

__global__ void FpTestAssociativeMul(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=0; i<TESTVALS; i++) {
        for (int j=0; j<TESTVALS; j++) {
            for (int k=0; k<TESTVALS; k++) {
                fp_t a, b, c;

                fp_cpy(a, testval[i]);  // x
                fp_cpy(b, testval[j]);  // y
                fp_cpy(c, testval[i]);  // x

                fp_mul(a, a, testval[j]);  // x * y
                fp_mul(a, a, testval[k]);  // (x * y) * z

                fp_mul(b, b, testval[k]);  // y * z
                fp_mul(c, c, b);           // x * (y * z)

                if (fp_neq(a, c)) {
                    pass = false;

                    printf("%d,%d,%d: FAILED: inconsistent result\n", i, j, k);
                    printf("x = "); fp_print(testval[i]);
                    printf("y = "); fp_print(testval[j]);
                    printf("z = "); fp_print(testval[k]);
                    printf("(x*y)*z = "); fp_print(a);
                    printf("x*(y*z) = "); fp_print(c);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

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
