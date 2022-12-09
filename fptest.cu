// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fp.cuh"
#include "fptest.cuh"

__managed__ testval_t testval[TESTVALS];

////////////////////////////////////////////////////////////

void init() {

    printf("%s\n", __func__);

    uint64_t
        p0 = 0xB9FEFFFFFFFFAAAB,
        p1 = 0x1EABFFFEB153FFFF,
        p2 = 0x6730D2A0F6B0F624,
        p3 = 0x64774B84F38512BF,
        p4 = 0x4B1BA7B6434BACD7,
        p5 = 0x1A0111EA397FE69A;

    int i = 0;

    for (int j=0; j<TESTVALS; j++) {
        testval[j].val[0] = 0;
        testval[j].val[1] = 0;
        testval[j].val[2] = 0;
        testval[j].val[3] = 0;
        testval[j].val[4] = 0;
        testval[j].val[5] = 0;
    }

    {
        testval_t t = { p0, p1, p2, p3, p4, p5 };
        testval[i] = t;
    }
    i++;

    {
        testval_t t = { ~p0, ~p1, ~p2, ~p3, ~p4, ~p5 };
        testval[i] = t;
    }
    i++;

    i++;    // The third value is 0

    for (int j=0; j<64; i++,j++) { testval[i].val[0] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i].val[1] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i].val[2] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i].val[3] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i].val[4] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i].val[5] = 1ULL << j; }

    for (int j=2; j<386; i++,j++) {
        testval[i].val[0] = ~testval[j].val[0];
        testval[i].val[1] = ~testval[j].val[1];
        testval[i].val[2] = ~testval[j].val[2];
        testval[i].val[3] = ~testval[j].val[3];
        testval[i].val[4] = ~testval[j].val[4];
        testval[i].val[5] = ~testval[j].val[5];
    }

    FILE *pf = fopen("/dev/urandom", "r");

    if (!pf)
        return;

    size_t result = fread(&testval[i], sizeof(testval_t), TESTVALS-i, pf);

    // Print all the random values
#if 0
    for (int j=i; j<TESTVALS; j++) {
        auto t = &testval[j];
        printf("0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
            t->val[5], t->val[4], t->val[3], t->val[2], t->val[1], t->val[0]);
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

    TEST(FpTestKAT);

    sleep(1);

    if (err != cudaSuccess)
        fprintf(stderr, "Error %d\n", err);

    TEST(FpTestCmp);
    TEST(FpTestMMA);
    /*
    TEST(FpTestMul);
    TEST(FpTestCopy);
    TEST(FpTestNeg);
    TEST(FpTestDouble);
    TEST(FpTestTriple);
    TEST(FpTestAdd);
    TEST(FpTestSub);
    TEST(FpTestSquare);
    TEST(FpTestMul);
    TEST(FpTestInv);

    TEST(FpTestReflexivity);
    TEST(FpTestSymmetry);
    TEST(FpTestAdditiveIdentity);
    TEST(FpTestMultiplicativeIdentity);
    TEST(FpTestAdditiveInverse);
    TEST(FpTestMultiplicativeInverse);
    TEST(FpTestCommutativeAdd);
    TEST(FpTestCommutativeMul);
    TEST(FpTestAssociativeAdd);
    TEST(FpTestAssociativeMul);
    TEST(FpTestDistributiveLeft);
    TEST(FpTestDistributiveRight);
*/
    sleep(1);

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
        fprintf(stderr, "Error %d\n", err);

    return err;
}

// vim: ts=4 et sw=4 si
