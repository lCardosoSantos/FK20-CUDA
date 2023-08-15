// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "g1.cuh"
#include "g1test.cuh"

__managed__ testval_t testval[TESTVALS];

////////////////////////////////////////////////////////////

/**
 * @brief initialization
 * 
 */
void init() {

    printf("%s\n", __func__);

    testinit();

    /*
    uint64_t t[2*TESTVALS];

    FILE *pf = fopen("/dev/urandom", "r");

    if (!pf)
        return;

    size_t result = fread(&testval[i], sizeof(testval_t), TESTVALS-i, pf);
    */
}

////////////////////////////////////////////////////////////
//Shorthand for testing a function, with an error check and timer
#define TEST(X) \
    start = clock(); \
    X <<<grid,block>>> (&testval[0]); \
    err = cudaDeviceSynchronize(); \
    end = clock(); \
    if (err != cudaSuccess) printf("Error %d (%s)\n", err, cudaGetErrorName(err)); \
    printf(" (%.2f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

////////////////////////////////////////////////////////////

/**
 * @brief Test for points in G1
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char **argv) {
    clock_t start, end;
    cudaError_t err;
#if 1
    dim3 block(1,1,1);
    dim3 grid(1,1,1);
#else
    dim3 block(32,8,1);
    dim3 grid(82,1,1);
#endif

    unsigned rows = 2;

    if (argc > 1)
        rows = atoi(argv[1]);

    if (rows < 1)
        rows = 1;

    if (rows > 512)
        rows = 512;

    init();

    TEST(G1TestKAT);
    TEST(G1TestFibonacci);
    TEST(G1TestDbl);

    G1TestFFT(rows);

    return err;
}

// vim: ts=4 et sw=4 si
