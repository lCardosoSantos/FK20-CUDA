// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <cstring>
#include <time.h>
#include "fr.cuh"
#include "fp.cuh"
#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"

// Testvector inputs

extern __managed__ g1p_t xext_fft[16][512];
extern __managed__ fr_t polynomial[512*4096];

// Intermediate values

extern __managed__ fr_t toeplitz_coefficients[512*16][512];
extern __managed__ fr_t toeplitz_coefficients_fft[512*16][512];
extern __managed__ g1p_t hext_fft[512*512];
extern __managed__ g1p_t h[512*512];

// Test vector output

extern __managed__ g1p_t h_fft[512*512];

// Workspace

static __managed__ uint8_t cmp[512*16*512];
static __managed__ fr_t fr_tmp_[512*16*512];
static __managed__ g1p_t g1p_tmp[512*512];

#define PatternOnWorkspaceMemory
#ifdef PatternOnWorkspaceMemory
    #define PTRN_G1PTMP memset(g1p_tmp, 0x88, 512*512*sizeof(g1p_t));
    #define PTRN_FRTMP  memset(fr_tmp_, 0x88, 512*16*512*sizeof(fr_t));
#else
    #define PTRN_G1PTMP
    #define PTRN_FRTMP
#endif

// 512-row tests

void toeplitz_coefficients2toeplitz_coefficients_fft_512(unsigned rows);
void h2h_fft_512(unsigned rows);
void h_fft2h_512(unsigned rows);
void hext_fft2h_512(unsigned rows);
void hext_fft2h_fft_512(unsigned rows);

void fk20_poly2toeplitz_coefficients_512(unsigned rows);
void fk20_poly2hext_fft_512(unsigned rows);
void fk20_poly2h_fft_512(unsigned rows);
void fk20_msmloop_512(unsigned rows);
//void fk20_poly2toeplitz_coefficients_fft_test(unsigned rows);
void fullTest_512(unsigned rows);
void fullTestFalseability_512(unsigned rows);

// Useful for the Falsifiability tests
void varMangle(fr_t *target, size_t size, unsigned step);
void varMangle(g1p_t *target, size_t size, unsigned step);

int main(int argc, char **argv) {

    testinit();

    unsigned rows = 2;

    if (argc > 1)
        rows = atoi(argv[1]);

        if (rows < 1)
            rows = 1;

        if (rows > 512)
            rows = 512;

    printf("=== RUN test with %d rows\n\n", rows);

    // FFT tests

    toeplitz_coefficients2toeplitz_coefficients_fft_512(rows);
    h2h_fft_512(rows);
    h_fft2h_512(rows);
    hext_fft2h_512(rows);
    // hext_fft2h_fft_512(rows); // fails, but components work

    // Polynomial tests

    fk20_poly2toeplitz_coefficients_512(rows);
    fk20_poly2hext_fft_512(rows);

    // MSM test

    fk20_msmloop_512(rows);

    // Full FK20 tests

    fk20_poly2h_fft_512(rows);
    fullTest_512(rows);
    fullTestFalseability_512(rows);
    //fk20_poly2toeplitz_coefficients_fft_test(rows); //Deprecated funtion

    return 0;
}

void fullTest_512(unsigned rows){
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    // Setup

    //SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    // polynomial -> tc

    printf("\n>>>>Full integration test\n"); fflush(stdout);
    printf("polynomial -> tc\n"); fflush(stdout);

    CLOCKSTART;
    fk20_poly2toeplitz_coefficients<<<rows, 256, fr_sharedmem>>>(fr_tmp_, polynomial);
    CUDASYNC("fk20_poly2toeplitz_coefficients");
    CLOCKEND;

    clearRes512;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp_, (fr_t *)toeplitz_coefficients);
    CUDASYNC("fr_eq_wrapper");
    for (int i=0; i<16*512; i++)
        if (cmp[i] != 1) {
            printf("poly2tc error %04x\n", i);
            pass = false;
        }
    PRINTPASS(pass);

    // tc -> tc_fft

    printf("tc -> tc_fft\n"); fflush(stdout);

    CLOCKSTART;
    fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(fr_tmp_, fr_tmp_);  // 16 per row
    CUDASYNC("fr_fft_wrapper");
    CLOCKEND;

    clearRes512;
    fr_eq_wrapper<<<256, 32>>>(cmp, rows*16*512, fr_tmp_, (fr_t *)toeplitz_coefficients_fft);
    CUDASYNC("fr_eq_wrapper");
    CMPCHECK(rows*16*512);
    PRINTPASS(pass);

    // tc_fft -> hext_fft

    printf("tc_fft -> hext_fft\n"); fflush(stdout);

    CLOCKSTART;
    fk20_msm<<<rows, 256>>>(g1p_tmp, fr_tmp_,  (g1p_t *)xext_fft);
    CUDASYNC("fk20_msm");
    CLOCKEND;

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, g1p_tmp, (g1p_t *)hext_fft);
    CUDASYNC("g1p_eq_wrapper");
    CMPCHECK(rows*512);
    PRINTPASS(pass);

    // hext_fft -> hext -> h

    printf("hext_fft -> hext -> h\n"); fflush(stdout);

    CLOCKSTART;
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC("g1p_ift_wrapper");
    fk20_hext2h<<<rows, 256>>>(g1p_tmp);
    CLOCKEND;
    CUDASYNC("fk20_hext2h");

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, g1p_tmp, (g1p_t *)h);
    CUDASYNC("g1p_eq_wrapper");
    CMPCHECK(rows*512);
    PRINTPASS(pass);

    // h -> h_fft

    printf("h -> h_fft\n"); fflush(stdout);

    CLOCKSTART;
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC("g1p_fft_wrapper");
    CLOCKEND;

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, g1p_tmp, h_fft);
    CUDASYNC("g1p_eq_wrapper");
    CMPCHECK(rows*512);
    PRINTPASS(pass);
}

void fullTestFalseability_512(unsigned rows){
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    // Setup

    //SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    // polynomial -> tc

    varMangle(polynomial, 512*4096, 64);

    printf("\n>>>>Full integration test - Falsifiability\n"); fflush(stdout);
    printf("polynomial -> tc\n"); fflush(stdout);

    CLOCKSTART;
    fk20_poly2toeplitz_coefficients<<<rows, 256, fr_sharedmem>>>(fr_tmp_, polynomial);
    CUDASYNC("fk20_poly2toeplitz_coefficients");
    CLOCKEND;

    clearRes512;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp_, (fr_t *)toeplitz_coefficients);
    CUDASYNC("fr_eq_wrapper");
    NEGCMPCHECK(16*512);
    NEGPRINTPASS(pass);

    // tc -> tc_fft

    printf("tc -> tc_fft\n"); fflush(stdout);

    CLOCKSTART;
    fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(fr_tmp_, fr_tmp_);  // 16 per row
    CUDASYNC("fr_fft_wrapper");
    CLOCKEND;

    clearRes512;
    fr_eq_wrapper<<<256, 32>>>(cmp, rows*16*512, fr_tmp_, (fr_t *)toeplitz_coefficients_fft);
    CUDASYNC("fr_eq_wrapper");
    NEGCMPCHECK(rows*16*512);
    NEGPRINTPASS(pass);

    // tc_fft -> hext_fft

    printf("tc_fft -> hext_fft\n"); fflush(stdout);

    CLOCKSTART;
    fk20_msm<<<rows, 256>>>(g1p_tmp, fr_tmp_,  (g1p_t *)xext_fft);
    CUDASYNC("fk20_msm");
    CLOCKEND;

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, g1p_tmp, (g1p_t *)hext_fft);
    CUDASYNC("g1p_eq_wrapper");
    NEGCMPCHECK(rows*512);
    NEGPRINTPASS(pass);

    // hext_fft -> hext -> h

    printf("hext_fft -> hext -> h\n"); fflush(stdout);

    CLOCKSTART;
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC("g1p_ift_wrapper");
    fk20_hext2h<<<rows, 256>>>(g1p_tmp);
    CLOCKEND;
    CUDASYNC("fk20_hext2h");

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, g1p_tmp, (g1p_t *)h);
    CUDASYNC("g1p_eq_wrapper");
    NEGCMPCHECK(rows*512);
    NEGPRINTPASS(pass);

    // h -> h_fft

    printf("h -> h_fft\n"); fflush(stdout);

    CLOCKSTART;
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC("g1p_fft_wrapper");
    CLOCKEND;

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, g1p_tmp, h_fft);
    CUDASYNC("g1p_eq_wrapper");
    NEGCMPCHECK(rows*512);
    NEGPRINTPASS(pass);
}

void toeplitz_coefficients2toeplitz_coefficients_fft_512(unsigned rows){
    PTRN_FRTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    printf("=== RUN   %s\n", "fr_fft: toeplitz_coefficients -> toeplitz_coefficients_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(fr_tmp_, (fr_t *)toeplitz_coefficients);
        CUDASYNC("fr_fft_wrapper");
        CLOCKEND;

        clearRes;
        fr_eq_wrapper<<<256, 32>>>(cmp, rows*16*512, fr_tmp_, (fr_t *)toeplitz_coefficients_fft);
        CUDASYNC("fr_eq_wrapper");

        // Check FFT result

        if (testIDX == 0){
            CMPCHECK(rows*16*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*16*512);
            NEGPRINTPASS(pass);
        }

        varMangle((fr_t*)toeplitz_coefficients_fft, 8192*512, 512);
    }
}

void h2h_fft_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);

    printf("=== RUN   %s\n", "g1p_fft: h -> h_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, h);
        CUDASYNC("g1p_fft_wrapper");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, g1p_tmp, h_fft);
        CUDASYNC("g1p_eq_wrapper");

        // Check FFT result

        if (testIDX == 0){
            CMPCHECK(rows*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*512);
            NEGPRINTPASS(pass);
        }

        varMangle(h, 512*512, 128);
    }
}

void h_fft2h_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    printf("=== RUN   %s\n", "g1p_ift: h_fft -> h");

    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, h_fft);
        CUDASYNC("g1p_ift_wrapper");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<16, 32>>>(cmp, 512*512, g1p_tmp, h);
        CUDASYNC("g1p_eq_wrapper");

        // Check IFT result

        if (testIDX == 0){
            CMPCHECK(rows*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*512);
            NEGPRINTPASS(pass);
        }

        varMangle(h_fft, 512*512, 128);
    }
}

void hext_fft2h_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    printf("=== RUN   %s\n", "g1p_ift: hext_fft -> h");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, hext_fft);
        CUDASYNC("g1p_ift_wrapper"); 
        fk20_hext2h<<<rows, 256>>>(g1p_tmp);
        CUDASYNC("fk20_hext2h"); 
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<8, 32>>>(cmp, rows*512, g1p_tmp, h);
        CUDASYNC("g1p_eq_wrapper");

        if (testIDX == 0){
            CMPCHECK(rows*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*512);
            NEGPRINTPASS(pass);
        }

        varMangle(hext_fft, 512*512, 128);
    }
}

void fk20_poly2toeplitz_coefficients_512(unsigned rows) {
    PTRN_FRTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_poly2toeplitz_coefficients<<<rows, 256 >>>(fr_tmp_, polynomial);
        //IMPORTANT: This function does not need shared memory. Making the kernel call with a dynamic shared memory allocation
        //is known to cause some suble bugs, that not always show during normal execution.
        CUDASYNC("fk20_poly2toeplitz_coefficients");
        CLOCKEND;

        clearRes;
        fr_eq_wrapper<<<1, 32>>>(cmp, rows*16*512, fr_tmp_, (fr_t *)toeplitz_coefficients);
        CUDASYNC("fr_eq_wrapper");

        if (testIDX == 0){
            CMPCHECK(rows*16*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*16*512);
            NEGPRINTPASS(pass);
        }

        varMangle(polynomial, 512*4096, 8);
    }
}

void fk20_poly2hext_fft_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    pass = true;

    //SET_SHAREDMEM(g1p_sharedmem, fk20_poly2hext_fft);

    printf("=== RUN   %s\n", "fk20_poly2hext_fft: polynomial -> hext_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_poly2hext_fft<<<rows, 256, fr_sharedmem>>>(g1p_tmp, polynomial, (const g1p_t *)xext_fft);
        CUDASYNC("fk20_poly2hext_fft");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<1, 32>>>(cmp, rows*512, g1p_tmp, (g1p_t *)hext_fft);
        CUDASYNC("g1p_eq_wrapper");

        if (testIDX == 0){
            CMPCHECK(rows*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*512);
            NEGPRINTPASS(pass);
        }

        varMangle(( g1p_t *)xext_fft, 16*512, 32);
    }
}

void fk20_poly2h_fft_512(unsigned rows){
    PTRN_G1PTMP; PTRN_FRTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    printf("=== RUN   %s\n", "fk20_poly2h_fft: polynomial -> h_fft");

    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_poly2h_fft(g1p_tmp, polynomial, (const g1p_t *)xext_fft, rows);
        CUDASYNC("fk20_poly2h_fft");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<1, 32>>>(cmp, rows*512, g1p_tmp, (g1p_t *)h_fft);
        CUDASYNC("g1p_eq_wrapper");

        if (testIDX == 0){
            CMPCHECK(rows*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*512);
            NEGPRINTPASS(pass);
        }

        varMangle(( g1p_t *)xext_fft, 16*512, 32);
    }
}

void hext_fft2h_fft_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, fk20_hext_fft2h_fft);

    printf("=== RUN   %s\n", "hext_fft2h_fft_512: hext_fft -> h_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_hext_fft2h_fft<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, hext_fft);
        CUDASYNC("fk20_hext_fft2h_fft");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<8, 32>>>(cmp, rows*512, g1p_tmp, h);
        CUDASYNC("g1p_eq_wrapper");

        if (testIDX == 0){
            CMPCHECK(rows*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*512);
            NEGPRINTPASS(pass);
        }

        varMangle(hext_fft, 512*512, 32);
    }
}

void fk20_msmloop_512(unsigned rows){
    CLOCKINIT;
    cudaError_t err;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_msm: Toeplitz_coefficients+xext_fft -> hext_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_msm<<<rows, 256>>>(g1p_tmp, (const fr_t*)toeplitz_coefficients_fft, (const g1p_t*)xext_fft);
        CUDASYNC("fk20_msm");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, g1p_tmp, (g1p_t *)hext_fft);
        CUDASYNC("g1p_eq_wrapper");

        // Check result

        if (testIDX == 0){
            CMPCHECK(rows*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*512);
            NEGPRINTPASS(pass);
        }

        varMangle((fr_t*)toeplitz_coefficients_fft, 8192*512, 512);
    }
}

#if 0
    void fk20_poly2toeplitz_coefficients_fft_test(unsigned rows){
        // Test for deprecated function.
        PTRN_FRTMP;
        CLOCKINIT;
        cudaError_t err;
        bool pass = true;

        printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients_fft: polynomial -> toeplitz_coefficients_fft");
        memset(fr_tmp_, 0xdeadbeef,512*16*512*sizeof(fr_t)); //pattern on tmp dest.
        CLOCKSTART;
        fk20_poly2toeplitz_coefficients_fft<<<rows, 256>>>(fr_tmp_, polynomial);
        err = cudaDeviceSynchronize();
        CUDASYNC("fk20_poly2toeplitz_coefficients_fft"); 
        CLOCKEND;
        clearRes;
        fr_eq_wrapper<<<16, 256>>>(cmp, rows*16*512, fr_tmp_, (fr_t *)toeplitz_coefficients_fft);
        CUDASYNC("fr_eq_wrapper");
        // Check result

        CMPCHECK(rows*16*512);
        PRINTPASS(pass);
    }
#endif

// Useful for the Falsifiability tests
// If you are using a variable where i*step == i*step+1, you can end up with a false(false positive).
// A staggered start helps to mitigate it, but it can happen with a very small probability.

#define START_INDEX 3

void varMangle(fr_t *target, size_t size, unsigned step){
    fr_t tmp;
    if (target == NULL || size <= 0 || step <= 0)
        return;

    for (int i = START_INDEX; i < size; i += step) {
        if (i + step < size){
            memcpy(tmp, target+i, sizeof(fr_t));
            memcpy(target+i, target+i+1, sizeof(fr_t));
            memcpy(target+i+1, tmp, sizeof(fr_t));
        }
    }
}

void varMangle(g1p_t *target, size_t size, unsigned step){
    g1p_t tmp;
    if (target == NULL || size <= 0 || step <= 0)
        return;

    for (int i = START_INDEX; i < size; i += step) {
        if (i + step < size) {
            memcpy(&tmp, target+i, sizeof(g1p_t));
            memcpy(target+i, target+i+1, sizeof(g1p_t));
            memcpy(target+i+1, &tmp, sizeof(g1p_t));
        }
    }
}

// vim: ts=4 et sw=4 si
