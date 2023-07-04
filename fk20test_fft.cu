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

void FK20TestFFT() {
    printf(">>>> FFT tests\n");

    toeplitz_coefficients2toeplitz_coefficients_fft(toeplitz_coefficients, toeplitz_coefficients_fft); 
    h2h_fft(h, h_fft);
    h_fft2h(h_fft, h);
    hext_fft2h(hext_fft, h);
    hext_fft2h_fft(hext_fft, h_fft);

}

void toeplitz_coefficients2toeplitz_coefficients_fft(fr_t toeplitz_coefficients_l[16][512], fr_t toeplitz_coefficients_fft_l[16][512]){
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    printf("=== RUN   %s\n", "fr_fft: toeplitz_coefficients -> toeplitz_coefficients_fft");
    memset(fr_tmp, 0xdeadbeef,16*512*sizeof(fr_t)); //pattern on tmp dest.
    start = clock();
    fr_fft_wrapper<<<16, 256, fr_sharedmem>>>(fr_tmp, (fr_t *)toeplitz_coefficients_l);
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

    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft_l);

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
}

void h2h_fft(g1p_t h_l[512], g1p_t h_fft_l[512]){
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    err = cudaFuncSetAttribute(g1p_fft_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "g1p_fft: h -> h_fft");
    memset(g1p_tmp,0xdeadbeef,512*sizeof(g1p_t)); //pattern on tmp dest

    start = clock();
    g1p_fft_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, h_l);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess) printf("Error g1p_fft_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "g1p_eq_wrapper", cmp, 512, g1p_tmp, h_fft); fflush(stdout);

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_fft_l);

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
}

void h_fft2h(g1p_t h_fft_l[512], g1p_t h_l[512]){
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    err = cudaFuncSetAttribute(g1p_ift_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "g1p_ift: h_fft -> h");
    memset(g1p_tmp,0xdeadbeef,512*sizeof(g1p_t)); //pattern on tmp dest
    start = clock();
    g1p_ift_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, h_fft_l);
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

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_l);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Check IFT result

    for (int i=0; pass && i<512; i++)
        if (cmp[i] != 1) {
            printf("IFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);
}

void hext_fft2h(g1p_t hext_fft_l[512], g1p_t h_l[512]){
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    err = cudaFuncSetAttribute(g1p_ift_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "g1p_ift: hext_fft -> h");
    memset(g1p_tmp,0xdeadbeef,512*sizeof(g1p_t)); //pattern on tmp dest
    start = clock();
    g1p_ift_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, hext_fft_l);
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

    g1p_eq_wrapper<<<8, 32>>>(cmp, 256, g1p_tmp, h_l);    // Note: h, not hext, hence 256, not 512

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Check IFT result

    for (int i=0; pass && i<256; i++)
        if (cmp[i] != 1) {
            printf("IFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);
}


// vim: ts=4 et sw=4 si
void hext_fft2h_fft(g1p_t hext_fft_l[512], g1p_t h_fft_l[512]){
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    err = cudaFuncSetAttribute(fk20_hext_fft2h_fft, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "fk20_hext_fft2h_fft: hext_fft -> h_fft");
    memset(g1p_tmp,0xdeadbeef,512*sizeof(g1p_t)); //pattern on tmp dest

    start = clock();
    fk20_hext_fft2h_fft<<<1, 256, g1p_sharedmem>>>(g1p_tmp, hext_fft_l);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess) printf("Error fk20_hext_fft2h_fft: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "g1p_eq_wrapper", cmp, 512, g1p_tmp, h_fft); fflush(stdout);

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_fft_l);

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
}