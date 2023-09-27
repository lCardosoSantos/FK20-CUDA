// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_TEST_CUH
#define FP_TEST_CUH

#include <stdio.h>

#include "test.h"
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
TESTFUN(FpTestMulConstPTX);
TESTFUN(FpTestAdd);
TESTFUN(FpTestAddPTX);
TESTFUN(FpTestSub);
TESTFUN(FpTestSubPTX);
TESTFUN(FpTestSqr);
TESTFUN(FpTestSqrPTX);
TESTFUN(FpTestMul);
TESTFUN(FpTestMulPTX);
TESTFUN(FpTestInv);
TESTFUN(FpTestMMA);

// Quadratic

TESTFUN(FpTestSqr2);
TESTFUN(FpTestSqr2PTX);
TESTFUN(FpTestCommutativeAdd);
TESTFUN(FpTestCommutativeAddPTX);
TESTFUN(FpTestCommutativeMul);
TESTFUN(FpTestCommutativeMulPTX);

// Cubic

TESTFUN(FpTestAssociativeAdd);
TESTFUN(FpTestAssociativeAddPTX);
TESTFUN(FpTestAssociativeMul);
TESTFUN(FpTestAssociativeMulPTX);
TESTFUN(FpTestAddDistributiveLeft);
TESTFUN(FpTestAddDistributiveLeftPTX);
TESTFUN(FpTestAddDistributiveRight);
TESTFUN(FpTestAddDistributiveRightPTX);
TESTFUN(FpTestSubDistributiveLeft);
TESTFUN(FpTestSubDistributiveLeftPTX);
TESTFUN(FpTestSubDistributiveRight);
TESTFUN(FpTestSubDistributiveRightPTX);


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
