// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos


// This is a file used mainly as a dummy target for profiling, and it is not part
// of the FK20.
// This file is not guaranteed to be up to date.

// Known-good values generated by the Python implementation

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_profiler_api.h> 

#include "fr.cuh"
#include "fk20.cuh"
#include "g1.cuh"
#include "test.h"

extern __managed__ fr_t polynomial[4096];
extern __managed__ g1p_t setup[4097];
extern __managed__ g1p_t xext_fft[16][512];
extern __managed__ fr_t toeplitz_coefficients[16][512];
extern __managed__ fr_t toeplitz_coefficients_fft[16][512];
extern __managed__ g1p_t hext_fft[512];
extern __managed__ g1p_t h[512];
extern __managed__ g1p_t h_fft[512];

/**************************** Workspace variables *****************************/

static fr_t  *b_polynomial = NULL; //min[4096]; max[512*4096]
static g1p_t *b_xext_fft = NULL; //min[16][512]; max[16][512];
static fr_t  *b_toeplitz_coefficients = NULL; //min[16][512]; max [512*16][512];
static fr_t  *b_toeplitz_coefficients_fft = NULL; //min[16][512]; max [512*16][512];
static g1p_t *b_hext_fft = NULL; //min[512]; max [512*512];
static g1p_t *b_h = NULL; //min[512]; max [512*512];
static g1p_t *b_h_fft = NULL; //min[512]; max [512*512];

// Result pointers
static fr_t  *b_fr_tmp;
static g1p_t *b_g1p_tmp;

// The necessary shared memory is larger than what we can allocate statically, hence it is
// allocated dynamically in the kernel call. We set the maximum allowed size using this macro.
#define SET_SHAREDMEM(SZ, FN)                                                                                          \
    err = cudaFuncSetAttribute(FN, cudaFuncAttributeMaxDynamicSharedMemorySize, SZ);                                   \
    cudaDeviceSynchronize();                                                                                           \
    if (err != cudaSuccess)                                                                                            \
        printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

/**
 * @brief Write NCOPIES copies of SRC[SIZE] into DEST,
 *
 */
#define COPYMANY(DEST, SRC, SIZE, NCOPIES, TYPE)                                                     \
        for(int counter=0; counter<NCOPIES; counter++) memcpy(DEST+counter*SIZE, SRC, SIZE*sizeof(TYPE));

// Synchronizes the device, making sure that the kernel has finished executing.
// Checks for any errors, and reports if errors are found.
#define CUDASYNC(fmt, ...)                                                                                             \
    err = cudaDeviceSynchronize();                                                                                     \
    if (err != cudaSuccess)                                                                                            \
    printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)

/********************************* Prototypes *********************************/
void setupMemory(unsigned rows);
void freeMemory();


int main(int argc, char** argv){
    unsigned rows=32;

    if (argc > 1) rows = atoi(argv[1]);

    printf("running with %d rows\n", rows);

    cudaError_t err;
    setupMemory(rows);

    SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    fk20_poly2toeplitz_coefficients<<<rows, 256>>>(b_fr_tmp, b_polynomial);
    CUDASYNC("1");
    fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(b_fr_tmp, b_fr_tmp);
    CUDASYNC("2");
        cudaProfilerStart();
        fk20_msm<<<rows, 256>>>(b_g1p_tmp, b_fr_tmp,  (g1p_t *)xext_fft);
        cudaProfilerStop();
    CUDASYNC("3");
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(b_g1p_tmp, b_g1p_tmp);
    CUDASYNC("4");
    fk20_hext2h<<<rows, 256>>>(b_g1p_tmp);
    CUDASYNC("5");
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(b_g1p_tmp, b_g1p_tmp);
    CUDASYNC("6");

    
    return 0;
}

void setupMemory(unsigned rows){
    // Allocate memory and copy relevant data from the test vector
    // check, error on more than 193 rows
    cudaError_t err;
    #define MALLOCSYNC(fmt, ...) \
        if (err != cudaSuccess)                                                                                            \
        printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)

    err = cudaMallocManaged(&b_polynomial, rows*4096*sizeof(fr_t));
          MALLOCSYNC("b_polynomial");
    err = cudaMallocManaged(&b_xext_fft, 16*512*sizeof(g1p_t)); // size not dependant on number of rows
          MALLOCSYNC("id");
    // err = cudaMallocManaged(&b_toeplitz_coefficients, rows*16*512*sizeof(fr_t));
    //       MALLOCSYNC("id");
    // err = cudaMallocManaged(&b_toeplitz_coefficients_fft, rows*16*512*sizeof(fr_t));
    //       MALLOCSYNC("id");
    // err = cudaMallocManaged(&b_hext_fft, rows*512*sizeof(g1p_t));
    //       MALLOCSYNC("b_hext_fft");
    // err = cudaMallocManaged(&b_h, rows*512*sizeof(g1p_t));
    //       MALLOCSYNC("id");
    // err = cudaMallocManaged(&b_h_fft, rows*512*sizeof(g1p_t));
    //       MALLOCSYNC("b_h_fft");

    err = cudaMallocManaged(&b_g1p_tmp, rows*512*sizeof(g1p_t));
          MALLOCSYNC("b_g1p_tmp");
    err = cudaMallocManaged(&b_fr_tmp, rows*16*512*sizeof(fr_t));
          MALLOCSYNC("b_fr_tmp");


    // Copy data
    COPYMANY(b_polynomial, polynomial, 4096, rows, fr_t);
    COPYMANY(b_xext_fft, xext_fft, 16*512, 1, g1p_t);
    // COPYMANY(b_toeplitz_coefficients, toeplitz_coefficients, 16*512, rows, fr_t);
    // COPYMANY(b_toeplitz_coefficients_fft, toeplitz_coefficients_fft, 16*512, rows, fr_t);
    // COPYMANY(b_hext_fft, hext_fft, 512, rows, g1p_t);
    // COPYMANY(b_h, h, 512, rows, g1p_t);
    // COPYMANY(b_h_fft, h_fft, 512, rows, g1p_t);


    printf("Memory setup done!\n");
}

void freeMemory(){
    // No worries about freeing a NULL pointer, that check is done by cudaFree
    cudaFree(b_polynomial);
    cudaFree(b_xext_fft);
    cudaFree(b_toeplitz_coefficients);
    cudaFree(b_toeplitz_coefficients_fft);
    cudaFree(b_hext_fft);
    cudaFree(b_h);
    cudaFree(b_h_fft);
    printf("Allocated memory freed");
}
