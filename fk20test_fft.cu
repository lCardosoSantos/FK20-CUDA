// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>
#include <time.h>

#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"
#include "fk20_testvector.cuh"
#include "test.inc"

static __managed__ uint8_t cmp[16 * 512];
static __managed__ fr_t fr_tmp[16 * 512];
static __managed__ g1p_t g1p_tmp[512];

void FK20TestFFT() {
    printf(">>>> fft Tests\n");
    toeplitz_coefficients2toeplitz_coefficients_fft(toeplitz_coefficients, toeplitz_coefficients_fft); 
    h2h_fft(h, h_fft);
    h_fft2h(h_fft, h);
    hext_fft2h(hext_fft, h);
    hext_fft2h_fft(hext_fft, h_fft);

}

void toeplitz_coefficients2toeplitz_coefficients_fft(fr_t toeplitz_coefficients_l[16][512],
                                                     fr_t toeplitz_coefficients_fft_l[16][512]) {
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    printf("=== RUN   %s\n", "fr_fft: toeplitz_coefficients -> toeplitz_coefficients_fft");
    memset(fr_tmp, 0xAA, 16 * 512 * sizeof(fr_t)); // pattern on tmp dest.
    CLOCKSTART;

    fr_fft_wrapper<<<16, 256, fr_sharedmem>>>(fr_tmp, (fr_t *)(toeplitz_coefficients_l));
    CUDASYNC("fr_fft_wrapper");
    CLOCKEND;
    clearRes;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16 * 512, fr_tmp, (fr_t *)toeplitz_coefficients_fft_l);
    CUDASYNC("fr_eq_wrapper");
    CMPCHECK(16 * 512)
    PRINTPASS(pass);
}

void h2h_fft(g1p_t h_l[512], g1p_t h_fft_l[512]) {
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper)
    printf("=== RUN   %s\n", "g1p_fft: h -> h_fft");
    memset(g1p_tmp, 0xAA, 512 * sizeof(g1p_t)); // pattern on tmp dest

    CLOCKSTART;
    g1p_fft_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, h_l);
    CUDASYNC("g1p_fft_wrapper");
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_fft_l);
    CUDASYNC("g1p_eq_wrapper");

    CMPCHECK(512)
    PRINTPASS(pass);
}

void h_fft2h(g1p_t h_fft_l[512], g1p_t h_l[512]) {
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper)
    printf("=== RUN   %s\n", "g1p_ift: h_fft -> h");
    memset(g1p_tmp, 0xAA, 512 * sizeof(g1p_t)); // pattern on tmp dest
    CLOCKSTART;
    g1p_ift_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, h_fft_l);
    CUDASYNC("g1p_ift_wrapper");
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_l);
    CUDASYNC("g1p_eq_wrapper");
    // Check IFT result
    CMPCHECK(512)
    PRINTPASS(pass);
}

void hext_fft2h(g1p_t hext_fft_l[512], g1p_t h_l[512]){
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper)
    printf("=== RUN   %s\n", "g1p_ift: hext_fft -> h");
    memset(g1p_tmp,0xAA,512*sizeof(g1p_t)); //pattern on tmp dest
    CLOCKSTART;
    g1p_ift_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, hext_fft_l);
    CUDASYNC("g1p_ift_wrapper"); 
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<8, 32>>>(cmp, 256, g1p_tmp, h_l);    // Note: h, not hext, hence 256, not 512
    CUDASYNC("g1p_eq_wrapper");
    // Check IFT result

    CMPCHECK(256)
    PRINTPASS(pass);
}


void hext_fft2h_fft(g1p_t hext_fft_l[512], g1p_t h_fft_l[512]){
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, fk20_hext_fft2h_fft)
    printf("=== RUN   %s\n", "fk20_hext_fft2h_fft: hext_fft -> h_fft");
    memset(g1p_tmp,0x88,512*sizeof(g1p_t)); //pattern on tmp dest

    CLOCKSTART;
    fk20_hext_fft2h_fft<<<1, 256, g1p_sharedmem>>>(g1p_tmp, hext_fft_l);
    CUDASYNC("fk20_hext_fft2h_fft"); 
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_fft_l);
    CUDASYNC("g1p_eq_wrapper");

    // Check FFT result

    CMPCHECK(512)
    PRINTPASS(pass);
}
// vim: ts=4 et sw=4 si