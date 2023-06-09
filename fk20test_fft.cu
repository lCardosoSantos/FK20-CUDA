// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "g1.cuh"
#include "fk20test.cuh"
#include "fk20_testvector.cuh"

static __managed__ uint8_t cmp[16*512];
static __managed__ fr_t fr_tmp[16*512];
static __managed__ g1p_t g1p_tmp[512];

void FK20TestFFT() {

    const size_t g1p_sharedmem = 96*1024; //512*3*6*8; // 512 points * 3 residues/point * 6 words/residue * 8 bytes/word = 72 KiB
    const size_t fr_sharedmem = 512*4*8; // 512 residues * 4 words/residue * 8 bytes/word = 16 KiB
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    err = cudaFuncSetAttribute(g1p_fft_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    err = cudaFuncSetAttribute(g1p_ift_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    //////////////////////////////////////////////////
#if 0
    printf("=== RUN   %s\n", "fr_fft: toeplitz_coefficients -> toeplitz_coefficients_fft");
    start = clock();
    fr_fft_wrapper<<<16, 256, fr_sharedmem>>>(fr_tmp, (fr_t *)toeplitz_coefficients);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fr_fft_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<16*512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "fr_eq_wrapper", cmp, 512, fr_tmp, h_fft); fflush(stdout);

    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Check FFT result

    for (int i=0; pass && i<16*512; i++)
        if (cmp[i] != 1) {
            printf("FFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);
    //////////////////////////////////////////////////
#endif 
    printf("=== RUN   %s\n", "g1p_fft: h -> h_fft");

    start = clock();
    g1p_fft_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, h);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess) printf("Error g1p_fft_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "g1p_eq_wrapper", cmp, 512, g1p_tmp, h_fft); fflush(stdout);

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("Error g1p_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Check FFT result

    for (int i=0; pass && i<512; i++)
        if (cmp[i] != 1) {
            printf("FFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);

    #if 0
    //////////////////////////////////////////////////

    pass = true;

    printf("=== RUN   %s\n", "g1p_ift: h_fft -> h");

    start = clock();
    g1p_ift_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, h_fft);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error g1p_ift_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "g1p_eq_wrapper", cmp, 512, g1p_tmp, h); fflush(stdout);

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Check IFT result

    for (int i=0; pass && i<512; i++)
        if (cmp[i] != 1) {
            printf("IFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);

    //////////////////////////////////////////////////

    pass = true;

    printf("=== RUN   %s\n", "g1p_ift: hext_fft -> h");

    start = clock();
    g1p_ift_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, hext_fft);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error g1p_ift_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "g1p_eq_wrapper", cmp, 512, g1p_tmp, h); fflush(stdout);

    g1p_eq_wrapper<<<8, 32>>>(cmp, 256, g1p_tmp, h);    // Note: h, not hext, hence 256, not 512

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Check IFT result

    for (int i=0; pass && i<256; i++)
        if (cmp[i] != 1) {
            printf("IFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);
#endif
    //////////////////////////////////////////////////
}

// vim: ts=4 et sw=4 si
