// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "g1.cuh"
#include "test.h"
// #include "g1test.cuh"

#include "g1p_ptx.cuh"

#define TESTVALS 256

typedef struct {
    uint64_t val[22];
} testval_t;

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
//shortcut for kernel declaration
__global__ void G1_ADD_PTX(testval_t *testval);
__global__ void G1_SUB_PTX(testval_t *testval);
__global__ void G1_DBL_PTX(testval_t *testval);
__global__ void G1_ADDSUB_PTX(testval_t *testval);


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

    if (rows > 512)
        rows = 512;

    init();

    TEST(G1_ADD_PTX);
    TEST(G1_SUB_PTX);
    TEST(G1_DBL_PTX);
    TEST(G1_ADDSUB_PTX);


    return err;
}

__global__ void G1_ADD_PTX(testval_t *testval){
    printf("== TEST %s NOT IMPLEMENTED\n", __func__);

}

__global__ void G1_SUB_PTX(testval_t *testval){
    printf("== TEST %s NOT IMPLEMENTED\n", __func__);

}

__global__ void G1_DBL_PTX(testval_t *testval){
    printf("== TEST %s \n", __func__);
    bool pass = true;
    size_t count = 0;

    g1p_t out1, out0, in1, in0;
    g1p_t p, q, u, v;

    g1p_gen(p);
    g1p_gen(in0);

    g1p_print("cuda p = ", p);
    g1p_print("ptxm in0= ", in0); 
    printf("\n\n");

    for (int i=0; pass&i<20000; i++){
        g1p_dbl(p); 
        g1m(OP_DBL, out0, in0, in0, in0);

        if(g1p_neq(p, in0)){
            pass = false;
            printf("%d: FAILED\n", i);
            printf("FAILED\n" );
            g1p_print("cuda = ", p);
            g1p_print("ptxm = ", in0);
            pass = false;
        }

        //g1p_cpy(in0, out0);
        ++count;
    }

    if (!pass || (blockIdx.x | blockIdx.y | blockIdx.z | threadIdx.x | threadIdx.y | threadIdx.z) == 0){
        printf("%ld tests\n", count);
        PRINTPASS(pass);
    }

    }

__global__ void G1_ADDSUB_PTX(testval_t *testval){
    printf("== TEST %s NOT IMPLEMENTED\n", __func__);

}
