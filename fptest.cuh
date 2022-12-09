// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FP_TEST_CUH
#define FP_TEST_CUH

#include <stdio.h>

#include "fp.cuh"

#define TESTVALS 896

typedef struct {
    uint64_t val[6];
} testval_t;

#define TESTFUN(X) extern __global__ void X(testval_t *testval)

TESTFUN(FpTestKAT);
TESTFUN(FpTestCmp);
TESTFUN(FpTestMMA);
TESTFUN(FpTestMul);
TESTFUN(FpTestCopy);
TESTFUN(FpTestEqNeq);
TESTFUN(FpTestReflexivity);
TESTFUN(FpTestSymmetry);
TESTFUN(FpTestAdditiveIdentity);
TESTFUN(FpTestMultiplicativeIdentity);
TESTFUN(FpTestAdditiveInverse);
TESTFUN(FpTestMultiplicativeInverse);
TESTFUN(FpTestCommutativeAdd);
TESTFUN(FpTestCommutativeMul);
TESTFUN(FpTestAssociativeAdd);
TESTFUN(FpTestAssociativeMul);
TESTFUN(FpTestDistributiveLeft);
TESTFUN(FpTestDistributiveRight);
TESTFUN(FpTestDouble);
TESTFUN(FpTestSquare);

#endif // FP_TEST_CUH

// vim: ts=4 et sw=4 si
