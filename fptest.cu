// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

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
        testval[j][0] = 0;
        testval[j][1] = 0;
        testval[j][2] = 0;
        testval[j][3] = 0;
        testval[j][4] = 0;
        testval[j][5] = 0;
    }

    {
        testval[i][0] = p0;
        testval[i][1] = p1;
        testval[i][2] = p2;
        testval[i][3] = p3;
        testval[i][4] = p4;
        testval[i][5] = p5;
    }
    i++;

    {
        testval[i][0] = ~p0;
        testval[i][1] = ~p1;
        testval[i][2] = ~p2;
        testval[i][3] = ~p3;
        testval[i][4] = ~p4;
        testval[i][5] = ~p5;
    }
    i++;

    i++;    // The third value is 0

    for (int j=0; j<64; i++,j++) { testval[i][0] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i][1] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i][2] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i][3] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i][4] = 1ULL << j; }
    for (int j=0; j<64; i++,j++) { testval[i][5] = 1ULL << j; }

    for (int j=2; j<386; i++,j++) {
        testval[i][0] = ~testval[j][0];
        testval[i][1] = ~testval[j][1];
        testval[i][2] = ~testval[j][2];
        testval[i][3] = ~testval[j][3];
        testval[i][4] = ~testval[j][4];
        testval[i][5] = ~testval[j][5];
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

    TEST(FpTestKAT);
    if (err != cudaSuccess) {
        return err;
    }

    if (level >= 1) {
        TEST(FpTestCmp);
        TEST(FpTestMulConst);
        TEST(FpTestAdd);
        TEST(FpTestSub);
        TEST(FpTestSqr);
        TEST(FpTestMul);
        TEST(FpTestInv);
    }

    if (level >= 2) {
        TEST(FpTestSqr2);
        TEST(FpTestCommutativeAdd);
//      TEST(FpTestCommutativeMul);
    }

    if (level >= 3) {
        TEST(FpTestAssociativeAdd);
//      TEST(FpTestAssociativeMul);
//      TEST(FpTestDistributiveLeft);
//      TEST(FpTestDistributiveRight);
        TEST(FpTestMMA);
    }

    /*
    TEST(FpTestCopy);
    TEST(FpTestNeg);
    TEST(FpTestDouble);
    TEST(FpTestTriple);
    TEST(FpTestAdd);
    TEST(FpTestSub);
    TEST(FpTestSquare);
    TEST(FpTestMul);

    TEST(FpTestReflexivity);
    TEST(FpTestSymmetry);
    TEST(FpTestAdditiveIdentity);
    TEST(FpTestMultiplicativeIdentity);
    TEST(FpTestAdditiveInverse);
    TEST(FpTestMultiplicativeInverse);
    */

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
        fprintf(stderr, "Error %d\n", err);

    return err;
}

// vim: ts=4 et sw=4 si
