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
    printf(">>>> Poly Tests\n");
    fk20_poly2toeplitz_coefficients_test(polynomial, toeplitz_coefficients);
    //fk20_poly2toeplitz_coefficients_fft_test(polynomial, toeplitz_coefficients_fft); //TODO: Fails
    fk20_poly2hext_fft_test(polynomial, xext_fft, hext_fft);
    fk20_poly2h_fft_test(polynomial, xext_fft, h_fft);
    fk20_msmloop(hext_fft, toeplitz_coefficients_fft, xext_fft);
}


void fk20_poly2toeplitz_coefficients_test(fr_t polynomial_l[4096], fr_t toeplitz_coefficients_l[16][512]){
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients");
    memset(fr_tmp, 0xdeadbeef,16*512*sizeof(fr_t)); //pattern on tmp dest.
    start = clock();
    fk20_poly2toeplitz_coefficients<<<1, 256>>>(fr_tmp, polynomial_l);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2toeplitz_coefficients: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<16*512; i++)
        cmp[i] = 0;

    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients_l);

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

void fk20_poly2hext_fft_test(fr_t polynomial_l[4096], g1p_t xext_fft_l[16][512], g1p_t hext_fft_l[512]){
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    err = cudaFuncSetAttribute(fk20_poly2hext_fft, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    //err = cudaFuncSetAttribute(fk20_poly2hext_fft, cudaFuncAttributeMaxDynamicSharedMemorySize, 512*16*sizeof(fr_t));
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "fk20_poly2hext_fft: polynomial -> hext_fft");
    memset(g1p_tmp,0xdeadbeef,512*sizeof(g1p_t)); //pattern on tmp dest
    start = clock();
    fk20_poly2hext_fft<<<1, 256, g1p_sharedmem>>>(g1p_tmp, polynomial_l, (const g1p_t *)xext_fft_l);
    //fk20_poly2hext_fft<<<1, 256>>>(g1p_tmp, polynomial_l, (const g1p_t *)xext_fft_l);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2hext_fft: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)hext_fft_l);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }

    PRINTPASS(pass);
}

void fk20_poly2h_fft_test(fr_t polynomial_l[4096], g1p_t xext_fft_l[16][512], g1p_t h_fft_l[512]){
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    err = cudaFuncSetAttribute(fk20_poly2h_fft, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "fk20_poly2h_fft: polynomial -> h_fft");
    memset(g1p_tmp,0xdeadbeef,512*sizeof(g1p_t)); //pattern on tmp dest
    start = clock();
    fk20_poly2h_fft<<<1, 256, g1p_sharedmem>>>(g1p_tmp, polynomial_l, (const g1p_t *)xext_fft_l); //this causes memory issues
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2h_fft: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)h_fft_l);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }

    PRINTPASS(pass);
}

void fk20_msmloop(g1p_t hext_fft_l[512], fr_t toeplitz_coefficients_fft_l[16][512], 
                  g1p_t xext_fft_l[16][512]){
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_msm: Toeplitz_coefficients+xext_fft -> hext_fft");
    memset(g1p_tmp,0xdeadbeef,512*sizeof(g1p_t)); //pattern on tmp dest
    start = clock();

    fk20_msm_xext_fftANDtoepliz_fft2hext_fft<<<1, 256>>>(g1p_tmp, (const fr_t*)toeplitz_coefficients_fft_l, (const g1p_t*)xext_fft_l);
    
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_msm: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)hext_fft_l);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }

    PRINTPASS(pass);
}

void fk20_poly2toeplitz_coefficients_fft_test(fr_t polynomial_l[4096], fr_t toeplitz_coefficients_fft_l[16][512]){
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients_fft: polynomial -> toeplitz_coefficients_fft");
    memset(fr_tmp, 0xdeadbeef,16*512*sizeof(fr_t)); //pattern on tmp dest.
    start = clock();
    fk20_poly2toeplitz_coefficients_fft<<<1, 256>>>(fr_tmp, polynomial_l);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2toeplitz_coefficients_fft: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<16*512; i++)
        cmp[i] = 0;

    fr_eq_wrapper<<<16, 256>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft_l);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<16*512; i++)
        if (cmp[i] != 1) {
            printf("poly2tc error %04x\n", i);
            pass = false;
            break;
        }

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
