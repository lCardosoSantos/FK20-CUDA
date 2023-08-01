// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"
#include "frtest.cuh"

__managed__ testval_t testval[TESTVALS];

////////////////////////////////////////////////////////////

void init() {

    testinit();

    uint64_t
        p0 = 0xFFFFFFFF00000001U,
        p1 = 0x53BDA402FFFE5BFEU,
        p2 = 0x3339D80809A1D805U,
        p3 = 0x73EDA753299D7D48U;

    int i = 0;

    for (int j=0; j<TESTVALS; j++) {
        testval[j][0] = 0;
        testval[j][1] = 0;
        testval[j][2] = 0;
        testval[j][3] = 0;
    }

    {
        testval[i][0] = p0;
        testval[i][1] = p1;
        testval[i][2] = p2;
        testval[i][3] = p3;
    }
    i++;

    {
        testval[i][0] = ~p0;
        testval[i][1] = ~p1;
        testval[i][2] = ~p2;
        testval[i][3] = ~p3;
    }
    i++;

    i++;    // The third value is 0

    for (int j=0; j<64; i++,j++) { testval[i][0] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i][1] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i][2] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i][3] = 1ULL << j; }

    for (int j=2; j<258; i++,j++) {
        testval[i][0] = ~testval[j][0];
        testval[i][1] = ~testval[j][1];
        testval[i][2] = ~testval[j][2];
        testval[i][3] = ~testval[j][3];
    }

    FILE *pf = fopen("/dev/urandom", "r");

    if (!pf)
        return;

    size_t result = fread(&testval[i], sizeof(testval_t), TESTVALS-i, pf);

    printf("Fixed/random test values: %d/%d\n", i, TESTVALS-i);
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

int main(int argc, char **argv) {
    clock_t start, end;
    cudaError_t err;

    int level = 0;

    if (argc > 1)
        level = atoi(argv[1]);

    init();

    dim3 block = 1;

    TEST(FrTestKAT);
    if (err != cudaSuccess) {
        return err;
    }
    FrTestFFT();

    if (level >= 1) {
        TEST(FrTestCmp);
        TEST(FrTestMulConst);
        TEST(FrTestSub);
        TEST(FrTestFibonacci);
    }

    if (level >= 2) {
        TEST(FrTestCommutativeAdd);
        TEST(FrTestCommutativeMul);
        TEST(FrTestAddSub);
    }

    if (level >= 3) {
        TEST(FrTestAssociativeAdd);
        TEST(FrTestAssociativeMul);
        TEST(FrTestAddDistributiveLeft);
        TEST(FrTestAddDistributiveRight);
        TEST(FrTestSubDistributiveLeft);
        TEST(FrTestSubDistributiveRight);
    }

/*
    TEST(FrTestCopy);
    TEST(FrTestNeg);
    TEST(FrTestMul);
    TEST(FrTestInv);

    TEST(FrTestReflexivity);
    TEST(FrTestSymmetry);
    TEST(FrTestAdditiveIdentity);
    TEST(FrTestMultiplicativeIdentity);
    TEST(FrTestAdditiveInverse);
    TEST(FrTestMultiplicativeInverse);
*/

    return 0;
}

// vim: ts=4 et sw=4 si
