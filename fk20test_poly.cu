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



void FK20TestPoly() {
    printf(">>>> Poly Tests\n");
    fk20_poly2toeplitz_coefficients_test(polynomial, toeplitz_coefficients);
    //fk20_poly2toeplitz_coefficients_fft_test(polynomial, toeplitz_coefficients_fft); //TODO: Not necessary, check fk20_poly2h_fft.cu
    fk20_poly2hext_fft_test(polynomial, xext_fft, hext_fft);
    fk20_msmloop(hext_fft, toeplitz_coefficients_fft, xext_fft);
    fk20_poly2h_fft_test(polynomial, xext_fft, h_fft);
    fullTest(); 

}

void fullTest(){ //TODO: How to do the falseability here?
    #define rows 1
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;
    // Setup
    SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    // polynomial -> tc
    printf("\n>>>>Full integration test\n"); fflush(stdout);
    printf("polynomial -> tc\n"); fflush(stdout);
    CLOCKSTART;
    fk20_poly2toeplitz_coefficients<<<rows, 256, fr_sharedmem>>>(fr_tmp, polynomial);
    CUDASYNC("fk20_poly2toeplitz_coefficients"); 
    CLOCKEND;
    clearRes;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients);
    CUDASYNC("fr_eq_wrapper"); 
    CMPCHECK(16*512);
    PRINTPASS(pass);

    // tc -> tc_fft
    printf("tc -> tc_fft\n"); fflush(stdout);
    CLOCKSTART;
    for(int i=0; i<16; i++){
        fr_fft_wrapper<<<rows, 256, fr_sharedmem>>>(fr_tmp+512*i, fr_tmp+512*i);  //needs to do 16 of those
    }
    CUDASYNC("fr_fft_wrapper"); 
    CLOCKEND;
    clearRes;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft);
    CUDASYNC("fr_eq_wrapper");
    CMPCHECK(16*512);
    PRINTPASS(pass);

    // tc_fft -> hext_fft
    printf("tc_fft -> hext_fft\n"); fflush(stdout);
    CLOCKSTART;
    fk20_msm<<<rows, 256>>>(g1p_tmp, fr_tmp,  (g1p_t *)xext_fft);
    CUDASYNC("fk20_msm");
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)hext_fft);
    CUDASYNC("g1p_eq_wrapper");
    CMPCHECK(512);
    PRINTPASS(pass);

    // hext_fft -> hext -> h
    printf("hext_fft -> hext -> h\n"); fflush(stdout);
    CLOCKSTART;
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC("g1p_ift_wrapper");
    fk20_hext2h<<<rows, 256>>>(g1p_tmp);
    CLOCKEND;
    CUDASYNC("fk20_hext2h");
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 256, g1p_tmp, (g1p_t *)h);
    CUDASYNC("g1p_eq_wrapper");
    CMPCHECK(256);
    PRINTPASS(pass);
    
    //h -> h_fft
    printf("h -> h_fft\n"); fflush(stdout);
    CLOCKSTART;
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC("g1p_fft_wrapper");
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_fft);
    CUDASYNC("g1p_eq_wrapper");
    CMPCHECK(512);
    PRINTPASS(pass);
#undef rows
}
void fullTestFalseability(){ //TODO: How to do the falseability here?
    #define rows 1
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;
    // Setup
    SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    // polynomial -> tc
    printf("\n>>>>Full integration test\n"); fflush(stdout);
    printf("polynomial -> tc\n"); fflush(stdout);
    CLOCKSTART;
    fk20_poly2toeplitz_coefficients<<<rows, 256, fr_sharedmem>>>(fr_tmp, polynomial);
    CUDASYNC("fk20_poly2toeplitz_coefficients"); 
    CLOCKEND;
    clearRes;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients);
    CUDASYNC("fr_eq_wrapper"); 
    NEGCMPCHECK(16*512);
    NEGPRINTPASS(pass);

    // tc -> tc_fft
    printf("tc -> tc_fft\n"); fflush(stdout);
    CLOCKSTART;
    for(int i=0; i<16; i++){
        fr_fft_wrapper<<<rows, 256, fr_sharedmem>>>(fr_tmp+512*i, fr_tmp+512*i);  //needs to do 16 of those
    }
    CUDASYNC("fr_fft_wrapper"); 
    CLOCKEND;
    clearRes;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft);
    CUDASYNC("fr_eq_wrapper");
    NEGCMPCHECK(16*512);
    NEGPRINTPASS(pass);

    // tc_fft -> hext_fft
    printf("tc_fft -> hext_fft\n"); fflush(stdout);
    CLOCKSTART;
    fk20_msm<<<rows, 256>>>(g1p_tmp, fr_tmp,  (g1p_t *)xext_fft);
    CUDASYNC("fk20_msm");
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)hext_fft);
    CUDASYNC("g1p_eq_wrapper");
    NEGCMPCHECK(512);
    NEGPRINTPASS(pass);

    // hext_fft -> hext -> h
    printf("hext_fft -> hext -> h\n"); fflush(stdout);
    CLOCKSTART;
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC("g1p_ift_wrapper");
    fk20_hext2h<<<rows, 256>>>(g1p_tmp);
    CLOCKEND;
    CUDASYNC("fk20_hext2h");
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 256, g1p_tmp, (g1p_t *)h);
    CUDASYNC("g1p_eq_wrapper");
    NEGCMPCHECK(256);
    NEGPRINTPASS(pass);
    
    //h -> h_fft
    printf("h -> h_fft\n"); fflush(stdout);
    CLOCKSTART;
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p_tmp, g1p_tmp);
    CUDASYNC("g1p_fft_wrapper");
    CLOCKEND;
    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, h_fft);
    CUDASYNC("g1p_eq_wrapper");
    NEGCMPCHECK(512);
    NEGPRINTPASS(pass);
#undef rows
}

void fk20_poly2toeplitz_coefficients_test(fr_t polynomial_l[4096], fr_t toeplitz_coefficients_l[16][512]){
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;
    
    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients");
    memset(fr_tmp, 0xAA,16*512*sizeof(fr_t)); //pattern on tmp dest.
    for(int testIDX=0; testIDX<=1; testIDX++){
        CLOCKSTART;
        fk20_poly2toeplitz_coefficients<<<1, 256>>>(fr_tmp, polynomial_l);
        CUDASYNC("fk20_poly2toeplitz_coefficients");
        CLOCKEND;
        clearRes;
        fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients_l);
        CUDASYNC("fr_eq_wrapper");
        // Check result
        if (testIDX == 0){
            CMPCHECK(16 * 512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(16*512);
            NEGPRINTPASS(pass);
        }
        
        varMangle((fr_t*)polynomial_l, 4096, 512);
    }
}

void fk20_poly2hext_fft_test(fr_t polynomial_l[4096], g1p_t xext_fft_l[16][512], g1p_t hext_fft_l[512]){
    cudaError_t err;
    CLOCKINIT;
    bool pass = true;

    SET_SHAREDMEM(g1p_sharedmem, fk20_poly2hext_fft)

    printf("=== RUN   %s\n", "fk20_poly2hext_fft: polynomial -> hext_fft");
    memset(g1p_tmp,0xAA,512*sizeof(g1p_t)); //pattern on tmp dest
    for(int testIDX=0; testIDX<=1; testIDX++){
        CLOCKSTART;
        fk20_poly2hext_fft<<<1, 256, g1p_sharedmem>>>(g1p_tmp, polynomial_l, (const g1p_t *)xext_fft_l);
        //fk20_poly2hext_fft<<<1, 256>>>(g1p_tmp, polynomial_l, (const g1p_t *)xext_fft_l);
        CUDASYNC("fk20_poly2hext_fft");
        CLOCKEND;
        clearRes;
        g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)hext_fft_l);
        CUDASYNC("g1p_eq_wrapper");

        // Check result
        if (testIDX == 0){
            CMPCHECK(512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(512);
            NEGPRINTPASS(pass);
        }
        varMangle(hext_fft_l, 512, 64);
    }
}

void fk20_poly2h_fft_test(fr_t polynomial_l[4096], g1p_t xext_fft_l[16][512], g1p_t h_fft_l[512]){
    cudaError_t err;
    CLOCKINIT;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_poly2h_fft: polynomial -> h_fft (full computation)");
    //memset(g1p_tmp,0x88,512*sizeof(g1p_t)); //pattern on tmp dest
    memset(g1p_tmp,0,512*sizeof(g1p_t)); //pattern on tmp dest
    memset(fr_tmp,0xAA,8192*sizeof(fr_t)); //pattern on tmp dest
    for(int testIDX=0; testIDX<=1; testIDX++){
        CLOCKSTART;
        fk20_poly2h_fft(g1p_tmp, polynomial_l, (const g1p_t *)xext_fft_l, 1); //this causes memory issues
        CUDASYNC("fk20_poly2h_fft");
        CLOCKEND;
        clearRes;

        g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)h_fft_l);
        CUDASYNC("g1p_eq_wrapper");

        // Check result
        if (testIDX == 0){
            CMPCHECK(512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(512);
            NEGPRINTPASS(pass);
        }
        varMangle(h_fft_l, 512, 64);
    }
}

void fk20_msmloop(g1p_t hext_fft_l[512], fr_t toeplitz_coefficients_fft_l[16][512], 
                  g1p_t xext_fft_l[16][512]){
    cudaError_t err;
    CLOCKINIT;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_msm: Toeplitz_coefficients+xext_fft -> hext_fft");
    memset(g1p_tmp,0x88,512*sizeof(g1p_t)); //pattern on tmp dest
    for(int testIDX=0; testIDX<=1; testIDX++){
        CLOCKSTART;
        fk20_msm<<<1, 256>>>(g1p_tmp, (const fr_t*)toeplitz_coefficients_fft_l, (const g1p_t*)xext_fft_l);
        CUDASYNC("fk20_msm");
        CLOCKEND;
        clearRes;

        g1p_eq_wrapper<<<16, 32>>>(cmp, 512, g1p_tmp, (g1p_t *)hext_fft_l);
        CUDASYNC("g1p_eq_wrapper");
        // Check result
                if (testIDX == 0){
                CMPCHECK(512)
                PRINTPASS(pass);
                }
            else{
                NEGCMPCHECK(512);
                NEGPRINTPASS(pass);
            }
            varMangle(hext_fft_l, 512, 64);
        }
}

//Deprecated function
/*
void fk20_poly2toeplitz_coefficients_fft_test(fr_t polynomial_l[4096], fr_t toeplitz_coefficients_fft_l[16][512]){
    cudaError_t err;
    CLOCKINIT;
    bool pass = true;

    SET_SHAREDMEM(g1p_sharedmem, fk20_poly2toeplitz_coefficients_fft);

    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients_fft: polynomial -> toeplitz_coefficients_fft");
    memset(fr_tmp, 0xAA,16*512*sizeof(fr_t)); //pattern on tmp dest.
    CLOCKSTART;
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
*/

// vim: ts=4 et sw=4 si
