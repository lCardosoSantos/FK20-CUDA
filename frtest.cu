// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fr.cuh"
#include "frtest.cuh"

__managed__ testval_t testval[TESTVALS];

////////////////////////////////////////////////////////////

void init() {

    uint64_t
        p0 = 0xFFFFFFFF00000001U,
        p1 = 0x53BDA402FFFE5BFEU,
        p2 = 0x3339D80809A1D805U,
        p3 = 0x73EDA753299D7D48U;

    int i = 0;

    for (int j=0; j<TESTVALS; j++) {
        testval[j].val[0] = 0;
        testval[j].val[1] = 0;
        testval[j].val[2] = 0;
        testval[j].val[3] = 0;
    }

    {
        testval_t t = { p0, p1, p2, p3 };
        testval[i] = t;
    }
    i++;

    {
        testval_t t = { ~p0, ~p1, ~p2, ~p3 };
        testval[i] = t;
    }
    i++;

    i++;    // The third value is 0

    for (int j=0; j<64; i++,j++) { testval[i].val[0] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i].val[1] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i].val[2] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i].val[3] = 1ULL << j; }

    for (int j=2; j<258; i++,j++) {
        testval[i].val[0] = ~testval[j].val[0];
        testval[i].val[1] = ~testval[j].val[1];
        testval[i].val[2] = ~testval[j].val[2];
        testval[i].val[3] = ~testval[j].val[3];
    }

    FILE *pf = fopen("/dev/urandom", "r");

    if (!pf)
        return;

    size_t result = fread(&testval[i], sizeof(testval_t), TESTVALS-i, pf);

    // Print all the random values
#if 0
    for (int j=i; j<TESTVALS; j++) {
        auto t = &testval[j];
        printf("0x%016lx%016lx%016lx%016lx\n",
            t->val[3], t->val[2], t->val[1], t->val[0]);
    }
#endif
}

////////////////////////////////////////////////////////////

#define TEST(X) \
    start = clock(); \
    X <<<1,block>>> (&testval[0]); \
    err = cudaDeviceSynchronize(); \
    end = clock(); \
    if (err != cudaSuccess) printf("Error %d\n", err); \
    printf(" (%.2f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

////////////////////////////////////////////////////////////

int main() {
    clock_t start, end;

    cudaError_t err;

    init();

    dim3 block(1,1,1);

    TEST(FrTestKAT);
    TEST(FrTestCmp);
/*
    TEST(FrTestCopy);
    TEST(FrTestNeg);
    TEST(FrTestDouble);
    TEST(FrTestTriple);
    TEST(FrTestAdd);
    TEST(FrTestSub);
    TEST(FrTestSquare);
    TEST(FrTestMul);
    TEST(FrTestInv);

    TEST(FrTestReflexivity);
    TEST(FrTestSymmetry);
    TEST(FrTestAdditiveIdentity);
    TEST(FrTestMultiplicativeIdentity);
    TEST(FrTestAdditiveInverse);
    TEST(FrTestMultiplicativeInverse);
    TEST(FrTestCommutativeAdd);
    TEST(FrTestCommutativeMul);
    TEST(FrTestAssociativeAdd);
    TEST(FrTestAssociativeMul);
    TEST(FrTestDistributiveLeft);
    TEST(FrTestDistributiveRight);
*/

    return 0;
}

// vim: ts=4 et sw=4 si
