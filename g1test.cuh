// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef G1_TEST_CUH
#define G1_TEST_CUH

#include "g1.cuh"

#define TESTVALS 256

typedef struct {
    uint64_t val[22];
} testval_t;

#define TESTFUN(X) extern __global__ void X(testval_t *testval)

TESTFUN(G1TestKAT);
TESTFUN(G1TestFibonacci);
TESTFUN(G1TestDbl);
TESTFUN(G1TestCmp);
TESTFUN(G1TestCopy);
TESTFUN(G1TestEqNeq);
TESTFUN(G1TestReflexivity);
TESTFUN(G1TestSymmetry);
TESTFUN(G1TestAdditiveIdentity);
TESTFUN(G1TestMultiplicativeIdentity);
TESTFUN(G1TestAdditiveInverse);
TESTFUN(G1TestMultiplicativeInverse);
TESTFUN(G1TestCommutativeAdd);
TESTFUN(G1TestCommutativeMul);
TESTFUN(G1TestAssociativeAdd);
TESTFUN(G1TestAssociativeMul);
TESTFUN(G1TestDistributiveLeft);
TESTFUN(G1TestDistributiveRight);
TESTFUN(G1TestDouble);
TESTFUN(G1TestSquare);

#endif // G1_TEST_CUH

// vim: ts=4 et sw=4 si
