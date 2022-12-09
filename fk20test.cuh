// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FK20_TEST_CUH
#define FK20_TEST_CUH

#include <stdio.h>

#include "g1.cuh"
#include "fk20.cuh"

/*
#define TESTVALS 256

typedef struct {
    fp_t in[256];
    fp_t out[256];
} fft_testval_t;
*/

//#define TESTFUN(X) extern __global__ void X(testval_t *testval)
#define TESTFUN(X) extern void X()

TESTFUN(FK20TimeKAT);
TESTFUN(FK20VerifyKAT);

#endif // FK20_TEST_CUH

// vim: ts=4 et sw=4 si
