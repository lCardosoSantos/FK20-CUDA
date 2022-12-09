// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FR_TEST_CUH
#define FR_TEST_CUH

#include <stdio.h>

#include "fr.cuh"

#define TESTVALS 768

typedef struct {
    uint64_t val[4];
} testval_t;

#define TESTFUN(X) extern __global__ void X(testval_t *testval)

TESTFUN(FrTestKAT);
TESTFUN(FrTestCmp);
TESTFUN(FrTestCopy);
TESTFUN(FrTestEqNeq);
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
TESTFUN(FrTestDistributiveLeft);
TESTFUN(FrTestDistributiveRight);
TESTFUN(FrTestDouble);
TESTFUN(FrTestSquare);

#endif // FR_TEST_CUH

// vim: ts=4 et sw=4 si
