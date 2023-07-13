// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>
#include <time.h>

#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"
#include "fk20_testvector.cuh"


static __managed__ uint8_t cmp[16*512];
static __managed__ fr_t fr_tmp[16*512];
static __managed__ g1p_t g1p_tmp[512];

void fullTest(){
    //TODO: Use defines on the other tests for simplification
#define CUDASYNC     err = cudaDeviceSynchronize(); \
                     if (err != cudaSuccess) printf("Error: %d (%s)\n", err, cudaGetErrorName(err))
#define SET_SHAREDMEM(SZ, FN) \
    err = cudaFuncSetAttribute(FN, cudaFuncAttributeMaxDynamicSharedMemorySize, SZ); \
    cudaDeviceSynchronize(); \
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

#define clearRes   for (int i=0; i<16*512; i++) cmp[i] = 0; \
                   pass=true;
#define rows 1

#define CLOCKINIT clock_t start, end
#define CLOCKSTART start=clock()
#define CLOCKEND end = clock();\
                 printf(" (%.1f ms)\n", (end - start) * (1000. / CLOCKS_PER_SEC))

    cudaError_t err;
    bool pass = true;
    CLOCKINIT;


    // Setup
    SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    //SET_SHAREDMEM(fr_sharedmem,  fk20_msm);
    //SET_SHAREDMEM(g1p_sharedmem, fk20_hext_fft2h_fft);  // function not being used?
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    // polynomial -> tc
    printf("\n>>>>FullTest\n"); fflush(stdout);
    printf("polynomial -> tc\n"); fflush(stdout);
    CLOCKSTART;
    fk20_poly2toeplitz_coefficients<<<rows, 256, fr_sharedmem>>>(fr_tmp, polynomial);
    CUDASYNC; 
    CLOCKEND;
    clearRes;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients);
    CUDASYNC; 
    for (int i=0; i<16*512; i++)
        if (cmp[i] != 1) {
            printf("poly2tc error %04x\n", i);
            pass = false;
        }
    PRINTPASS(pass);

    // tc -> tc_fft
    printf("tc -> tc_fft\n"); fflush(stdout);
    CLOCKSTART;
    for(int i=0; i<16; i++){
        fr_fft_wrapper<<<rows, 256, fr_sharedmem>>>(fr_tmp+512*i, fr_tmp+512*i);  //needs to do 16 of those
    }
    CUDASYNC; 
    CLOCKEND;
    clearRes;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft);
    CUDASYNC;
    for (int i=0; i<16*512; i++)
        if (cmp[i] != 1) {
            printf("tc2tcfft %d\n", i);
            pass = false;
            break;
        }
    PRINTPASS(pass);

    // tc_fft -> hext_fft
    printf("tc_fft -> hext_fft\n"); fflush(stdout);
    CLOCKSTART;
    fk20_msm<<<rows, 256>>>(g1p_tmp, fr_tmp,  (g1p_t *)xext_fft);
    CUDASYNC;
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)hext_fft);
    CUDASYNC;
    for (int i=0; i<512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }
    PRINTPASS(pass);

    // hext_fft -> hext -> h
    printf("hext_fft -> hext -> h\n"); fflush(stdout);
    CLOCKSTART;
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC;
    fk20_hext2h<<<rows, 256>>>(g1p_tmp);
    CLOCKEND;
    CUDASYNC;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 256, g1p_tmp, (g1p_t *)h);
    CUDASYNC;
    for (int i=0; i<256; i++)
        if (cmp[i] != 1) {
            pass = false;
        }
    PRINTPASS(pass);
    
    //h -> h_fft
    printf("h -> h_fft\n"); fflush(stdout);
    CLOCKSTART;
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC;
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_fft);
    CUDASYNC;
    for (int i=0; i<512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }
    PRINTPASS(pass);

}

void FK20TestPoly() {
    printf(">>>> Poly Tests\n");
    fk20_poly2toeplitz_coefficients_test(polynomial, toeplitz_coefficients);
    //fk20_poly2toeplitz_coefficients_fft_test(polynomial, toeplitz_coefficients_fft); //TODO: Not necessary, check fk20_poly2h_fft.cu
    fk20_poly2hext_fft_test(polynomial, xext_fft, hext_fft);
    fk20_msmloop(hext_fft, toeplitz_coefficients_fft, xext_fft);
    fk20_poly2h_fft_test(polynomial, xext_fft, h_fft);
    fullTest(); //TODO: Mainly for debugging, similar to fk20_poly2h_fft_test

}

void fk20_poly2toeplitz_coefficients_test(fr_t polynomial_l[4096], fr_t toeplitz_coefficients_l[16][512]){
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients");
    memset(fr_tmp, 0xAA,16*512*sizeof(fr_t)); //pattern on tmp dest.
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
    memset(g1p_tmp,0xAA,512*sizeof(g1p_t)); //pattern on tmp dest
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

    printf("=== RUN   %s\n", "fk20_poly2h_fft: polynomial -> h_fft (full computation)");
    //memset(g1p_tmp,0x88,512*sizeof(g1p_t)); //pattern on tmp dest
    memset(g1p_tmp,0,512*sizeof(g1p_t)); //pattern on tmp dest
    memset(fr_tmp,0xAA,8192*sizeof(fr_t)); //pattern on tmp dest
    start = clock();
    fk20_poly2h_fft(g1p_tmp, polynomial_l, (const g1p_t *)xext_fft_l, 1); //this causes memory issues
    err = cudaDeviceSynchronize();
    end = clock();
        //printf(__FILE__  " g1p_tmp \n");
        //WRITEU64STDOUT( g1p_tmp, 36);
        //
        //printf(__FILE__  " kat \n");
        //WRITEU64STDOUT( h_fft_l, 36);

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
    memset(g1p_tmp,0x88,512*sizeof(g1p_t)); //pattern on tmp dest
    start = clock();

    fk20_msm<<<1, 256>>>(g1p_tmp, (const fr_t*)toeplitz_coefficients_fft_l, (const g1p_t*)xext_fft_l);
    
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
    memset(fr_tmp, 0xAA,16*512*sizeof(fr_t)); //pattern on tmp dest.
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
