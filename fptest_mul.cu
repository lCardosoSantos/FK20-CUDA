// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fptest.cuh"

/**
 * @brief Test for the commutative property of addition
 * 
 * x*y == y*x
 * 
 * @param testval 
 * @return void 
 */
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
                fp_print("x = ",  testval[i]);
                fp_print("y = ",  testval[j]);
                fp_print("x*y = ",  x);
                fp_print("y*x = ",  y);
            }
            ++count;
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// 

/**
 * @brief Test for the associative property of multiplication
 * 
 * (x*y)*z == x*(y*z)
 * 
 * @param testval 
 * @return void 
 */
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
                    fp_print("x = ",  testval[i]);
                    fp_print("y = ",  testval[j]);
                    fp_print("z = ",  testval[k]);
                    fp_print("(x*y)*z = ",  a);
                    fp_print("x*(y*z) = ",  c);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

/**
 * @brief Multiplication test, using different values for different threads.
 * 
 * 
 * @param testval 
 * @return void 
 */
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
    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
