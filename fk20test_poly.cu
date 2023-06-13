// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"
#include "fk20_testvector.cuh"

static __managed__ uint8_t cmp[16*512];
static __managed__ fr_t fr_tmp[16*512];
static __managed__ g1p_t g1p_tmp[512];

void FK20TestPoly() {
    printf(">>>>");
    
    fk20_poly2toeplitz_coefficients_test();
    fk20_poly2hext_fft_test();
    fk20_poly2h_fft_test();
}


void fk20_poly2toeplitz_coefficients_test(){
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients");

    start = clock();
    fk20_poly2toeplitz_coefficients<<<1, 256>>>(fr_tmp, polynomial);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2toeplitz_coefficients: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<16*512; i++)
        cmp[i] = 0;

    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<16*512; i++)
        if (cmp[i] != 1) {
            printf("poly2tc error %04x\n", i);
            pass = false;
        }

    PRINTPASS(pass);
}

void fk20_poly2hext_fft_test(){
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    err = cudaFuncSetAttribute(fk20_poly2hext_fft, cudaFuncAttributeMaxDynamicSharedMemorySize, fr_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "fk20_poly2hext_fft: polynomial -> hext_fft");

    start = clock();
    fk20_poly2hext_fft<<<1, 256, g1p_sharedmem>>>(g1p_tmp, polynomial, (const g1p_t *)xext_fft);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2hext_fft: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)hext_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }

    PRINTPASS(pass);
}

void fk20_poly2h_fft_test(){
    char g1p_tmp_filename    [60];
    char h_fft_filename      [60];
    char xext_fft_filename   [60];
    char polynomial_filename [60];

    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    err = cudaFuncSetAttribute(fk20_poly2h_fft, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "fk20_poly2h_fft: polynomial -> h_fft");

    //for(int i=0; i<512; i++) 
    //    g1p_tmp[i] = {{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0}};

    //sprintf(g1p_tmp_filename   ,   "g1p_tmp.a%d.out", i);
    //sprintf(h_fft_filename     ,     "h_fft.a%d.out", i);
    //sprintf(xext_fft_filename  ,  "xext_fft.a%d.out", i);
    //sprintf(polynomial_filename,"polynomial.a%d.out", i);
    //
    //WRITEU64TOFILE(g1p_tmp_filename   , &g1p_tmp, 512*sizeof(g1p_t)/8);
    //WRITEU64TOFILE(h_fft_filename     , &h_fft, 512*sizeof(g1p_t)/8);
    //WRITEU64TOFILE(xext_fft_filename  , xext_fft, 16*512*sizeof(g1p_t)/8);
    //WRITEU64TOFILE(polynomial_filename, polynomial, 4096*sizeof(fr_t)/8);

    start = clock();
    fk20_poly2h_fft<<<1, 256, g1p_sharedmem>>>(g1p_tmp, polynomial, (const g1p_t *)xext_fft); //this causes memory issues
    err = cudaDeviceSynchronize();
    end = clock();

    //sprintf(g1p_tmp_filename   ,   "g1p_tmp.d%d.out", i);
    //sprintf(h_fft_filename     ,     "h_fft.d%d.out", i);
    //sprintf(xext_fft_filename  ,  "xext_fft.d%d.out", i);
    //sprintf(polynomial_filename,"polynomial.d%d.out", i);
    //
    //WRITEU64TOFILE(g1p_tmp_filename   , &g1p_tmp, 512*sizeof(g1p_t)/8);
    //WRITEU64TOFILE(h_fft_filename     , &h_fft, 512*sizeof(g1p_t)/8);
    //WRITEU64TOFILE(xext_fft_filename  , xext_fft, 16*512*sizeof(g1p_t)/8);
    //WRITEU64TOFILE(polynomial_filename, polynomial, 4096*sizeof(fr_t)/8);

    if (err != cudaSuccess)
        printf("Error fk20_poly2h_fft: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)h_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }

    PRINTPASS(pass);
}


// vim: ts=4 et sw=4 si
