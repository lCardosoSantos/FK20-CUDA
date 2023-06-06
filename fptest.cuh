// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FP_TEST_CUH
#define FP_TEST_CUH

#include <stdio.h>

#include "fp.cuh"

#define TESTVALS 800

#if (TESTVALS < 771)
# undef TESTVALS
# define TESTVALS 771
#endif

typedef fp_t testval_t;

#define TESTFUN(X) extern __global__ void X(testval_t *testval)

// Constant

TESTFUN(FpTestKAT);
TESTFUN(FpTestFibonacci);

// Linear

TESTFUN(FpTestCmp);
TESTFUN(FpTestMulConst);
TESTFUN(FpTestAdd);
TESTFUN(FpTestSub);
TESTFUN(FpTestSqr);
TESTFUN(FpTestMul);
TESTFUN(FpTestInv);
TESTFUN(FpTestMMA);

// Quadratic

TESTFUN(FpTestSqr2);
TESTFUN(FpTestCommutativeAdd);
TESTFUN(FpTestCommutativeMul);

// Cubic

TESTFUN(FpTestAssociativeAdd);
TESTFUN(FpTestAssociativeMul);
TESTFUN(FpTestAddDistributiveLeft);
TESTFUN(FpTestAddDistributiveRight);
TESTFUN(FpTestSubDistributiveLeft);
TESTFUN(FpTestSubDistributiveRight);

// Not implemented

TESTFUN(FpTestCopy);
TESTFUN(FpTestReflexivity);
TESTFUN(FpTestSymmetry);
TESTFUN(FpTestAdditiveIdentity);
TESTFUN(FpTestMultiplicativeIdentity);
TESTFUN(FpTestAdditiveInverse);
TESTFUN(FpTestMultiplicativeInverse);

#endif // FP_TEST_CUH

// vim: ts=4 et sw=4 si
