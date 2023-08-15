// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FR_TEST_CUH
#define FR_TEST_CUH

#include <stdio.h>

#include "test.h"
#include "fr.cuh"

#define TESTVALS 768

#if (TESTVALS < 515)
# undef TESTVALS
# define TESTVALS 515
#endif

typedef fr_t testval_t;

void FrTestFFT();

//shortcut for kernel declaration
#define TESTFUN(X) extern __global__ void X(testval_t *testval)

TESTFUN(FrTestKAT);
TESTFUN(FrTestFibonacci);
TESTFUN(FrTestCmp);
TESTFUN(FrTestMulConst);
TESTFUN(FrTestSub);
TESTFUN(FrTestAddSub);
TESTFUN(FrTestCopy);
TESTFUN(FrTestReflexivity);
TESTFUN(FrTestSymmetry);
TESTFUN(FrTestAdditiveIdentity);
TESTFUN(FrTestMultiplicativeIdentity);
TESTFUN(FrTestAdditiveInverse);
TESTFUN(FrTestMultiplicativeInverse);
TESTFUN(FrTestCommutativeAdd);
TESTFUN(FrTestCommutativeMul);
TESTFUN(FrTestAssociativeAdd);
TESTFUN(FrTestAssociativeMul);
TESTFUN(FrTestAddDistributiveLeft);
TESTFUN(FrTestAddDistributiveRight);
TESTFUN(FrTestSubDistributiveLeft);
TESTFUN(FrTestSubDistributiveRight);
TESTFUN(FrTestDouble);
TESTFUN(FrTestSquare);

#endif // FR_TEST_CUH

// vim: ts=4 et sw=4 si
