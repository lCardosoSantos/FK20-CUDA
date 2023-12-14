// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <cstring>
#include <time.h>
#include <unistd.h>
#include "fr.cuh"
#include "fp.cuh"
#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"

// Test vector inputs
g1p_t *xext_fft; //size [16][512];
fr_t  *polynomial; //size [512*4096];
// Intermediate values
fr_t  *toeplitz_coefficients; //size [512*16][512];
fr_t  *toeplitz_coefficients_fft; //size [512][16][512];
g1p_t *hext_fft; //size [512*512];
g1p_t *h; //size [512*512];
// Test vector output
g1p_t *h_fft; //size [512*512];

// Workspace

static __managed__ uint8_t cmp[512*16*512];
static __managed__ fr_t fr_tmp[512*16*512];
static __managed__ g1p_t g1p_tmp[512][512];
static __managed__ g1a_t xext_lut[16][512][256];

#define PatternOnWorkspaceMemory
#ifdef PatternOnWorkspaceMemory
    #define PTRN_G1PTMP memset(g1p_tmp, 0x88, 512*512*sizeof(g1p_t));
    #define PTRN_FRTMP  memset(fr_tmp, 0x88, 512*16*512*sizeof(fr_t));
#else
    #define PTRN_G1PTMP
    #define PTRN_FRTMP
#endif

// 512-row tests
void katInit();
void toeplitz_coefficients2toeplitz_coefficients_fft_512(unsigned rows);
void h2h_fft_512(unsigned rows);
void h_fft2h_512(unsigned rows);
void hext_fft2h_512(unsigned rows);
void hext_fft2h_fft_512(unsigned rows);
void hext_fft2h_fft_512_graph(unsigned rows);

void fk20_poly2toeplitz_coefficients_512(unsigned rows);
void fk20_poly2hext_fft_512(unsigned rows);
void fk20_poly2h_fft_512(unsigned rows);
void fk20_msmloop_512(unsigned rows);

void fk20_msmcomb_512(unsigned rows);
void fk20_msmcomb_512_graph(unsigned rows);

//void fk20_poly2toeplitz_coefficients_fft_test(unsigned rows);
void fullTest_512(unsigned rows);
void fullTestFalseability_512(unsigned rows);

// Useful for the Falsifiability tests
void varMangle(fr_t *target, size_t size, unsigned step);
void varMangle(g1p_t *target, size_t size, unsigned step);

/******************************************************************************/

/**
 * @brief Executes a many-row tests on FK20. Behavior is similar to fk20test.cu
 * but using many GPU blocks, each one executing one known-answer test. All tests
 * are different. KATS are statically linked in the binary.
 *
 * @param argc Command line argument cont
 * @param argv Command line argument values
 * @return int 0
 */
int main(int argc, char **argv) {
    testinit(); // setup functions here
    katInit();  // setup memory
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
    hext_fft2h_fft_512_graph(rows);
    // // hext_fft2h_fft_512(rows); //Deprecated function

    // Polynomial tests
    fk20_poly2toeplitz_coefficients_512(rows);
    fk20_poly2hext_fft_512(rows);

    // MSM test
    fk20_msmloop_512(rows);    
    fk20_msmcomb_512(rows);
    fk20_msmcomb_512_graph(rows);


    // Full FK20 tests
    fk20_poly2h_fft_512(rows);
    fullTest_512(rows);
    fullTestFalseability_512(rows);
    //fk20_poly2toeplitz_coefficients_fft_test(rows); //Deprecated function


}

/**
 * NOTE ON DEPRECATED FUNCTIONS
 *
 * In the main call, some tests are commented out, namely:
 * -hext_fft2h_fft_512
 * -fk20_poly2toeplitz_coefficients_fft_test
 * Those tests are regarding fk20 functions that execute more than one step in
 * a single kernel. They cover a unimplemented (possible) future optimization.
 *
 */
/******************************************************************************/

/**
 * @brief Executes many FK20 computations on a single row, with a check on
 * each step. A computation failure will not cause a cascade effect, eliminating
 * false-fails due to data dependencies.
 *
 * @param rows number of blocks in the range [1,512]
 */
void fullTest_512(unsigned rows){
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

    clearRes512;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients);
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
    fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(fr_tmp, fr_tmp);  // 16 per row
    CUDASYNC("fr_fft_wrapper");
    CLOCKEND;

    clearRes512;
    fr_eq_wrapper<<<256, 32>>>(cmp, rows*16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft);
    CUDASYNC("fr_eq_wrapper");
    CMPCHECK(rows*16*512);
    PRINTPASS(pass);

    // tc_fft -> hext_fft

    if(rows != 512){ 
        printf("tc_fft -> hext_fft\n"); fflush(stdout);
        printf("     WARNING: msm_comb runs with 512 rows only!\n");

        CLOCKSTART;
        fk20_msm<<<rows, 256>>>((g1p_t *)g1p_tmp, (fr_t *)toeplitz_coefficients_fft, (g1p_t *)xext_fft);
        CUDASYNC("fk20_msm");
        CLOCKEND;

        clearRes512;
        g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, (g1p_t *)hext_fft);
        CUDASYNC("g1p_eq_wrapper");
        CMPCHECK(rows*512);
        PRINTPASS(pass);
    }else{
        // Generate lookup tables for 8-way comb
        printf("Generating lookup tables for MSM with comb multiplication\n"); fflush(stdout);

        CLOCKSTART;
        fk20_msm_makelut<<<dim3(512, 16, 1), 1>>>(xext_lut, (const g1p_t (*)[512])(xext_fft));
        CUDASYNC("fk20_msm_makelut");
        CLOCKEND;

        // TODO?: fk20_msm_checklut<<<dim3(512, 16, 1), 1>>>(xext_lut, xext_fft);

        printf("MSM with comb multiplication\n"); fflush(stdout);

        CLOCKSTART;
        fk20_msm_comb<<<512, 256>>>(g1p_tmp, (const fr_t (*)[16][512])(toeplitz_coefficients_fft), xext_lut);
        CUDASYNC("fk20_msm_comb");
        CLOCKEND;

        clearRes512;
        g1p_eq_wrapper<<<16, 32>>>(cmp, 512*512, (g1p_t *)g1p_tmp, (g1p_t *)hext_fft);
        CUDASYNC("g1p_eq_wrapper");
        CMPCHECK(512*512);
        PRINTPASS(pass);
    }

    // hext_fft -> hext -> h

    printf("hext_fft -> hext -> h\n"); fflush(stdout);

    CLOCKSTART;
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>((g1p_t *)g1p_tmp, (g1p_t *)g1p_tmp);
    CUDASYNC("g1p_ift_wrapper");
    fk20_hext2h<<<rows, 256>>>((g1p_t *)g1p_tmp);
    CLOCKEND;
    CUDASYNC("fk20_hext2h");

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, (g1p_t *)h);
    CUDASYNC("g1p_eq_wrapper");
    CMPCHECK(rows*512);
    PRINTPASS(pass);

    // h -> h_fft

    printf("h -> h_fft\n"); fflush(stdout);

    CLOCKSTART;
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>((g1p_t *)g1p_tmp, (g1p_t *)g1p_tmp);
    CUDASYNC("g1p_fft_wrapper");
    CLOCKEND;

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, h_fft);
    CUDASYNC("g1p_eq_wrapper");
    CMPCHECK(rows*512);
    PRINTPASS(pass);
}

/**
 * @brief Similar to fullTest, but polynomial is has changes done to it. The
 * function checks for false-positive in the tests.
 *
 * polynomial is restored after execution.
 *
 * @param rows number of blocks in the range [1,512]
 */
void fullTestFalseability_512(unsigned rows){
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    // Setup

    SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    // polynomial -> tc

    varMangle(polynomial, 512*4096, 64);

    printf("\n>>>>Full integration test - Falsifiability\n"); fflush(stdout);
    printf("polynomial -> tc\n"); fflush(stdout);

    CLOCKSTART;
    fk20_poly2toeplitz_coefficients<<<rows, 256, fr_sharedmem>>>(fr_tmp, polynomial);
    CUDASYNC("fk20_poly2toeplitz_coefficients");
    CLOCKEND;

    clearRes512;
    fr_eq_wrapper<<<256, 32>>>(cmp, 16*512, fr_tmp, (fr_t *)toeplitz_coefficients);
    CUDASYNC("fr_eq_wrapper");
    NEGCMPCHECK(16*512);
    NEGPRINTPASS(pass);

    // tc -> tc_fft

    printf("tc -> tc_fft\n"); fflush(stdout);

    CLOCKSTART;
    fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(fr_tmp, fr_tmp);  // 16 per row
    CUDASYNC("fr_fft_wrapper");
    CLOCKEND;

    clearRes512;
    fr_eq_wrapper<<<256, 32>>>(cmp, rows*16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft);
    CUDASYNC("fr_eq_wrapper");
    NEGCMPCHECK(rows*16*512);
    NEGPRINTPASS(pass);

    // tc_fft -> hext_fft

    printf("tc_fft -> hext_fft\n"); fflush(stdout);

    CLOCKSTART;
    fk20_msm<<<rows, 256>>>((g1p_t *)g1p_tmp, fr_tmp,  (g1p_t *)xext_fft);
    CUDASYNC("fk20_msm");
    CLOCKEND;

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, (g1p_t *)hext_fft);
    CUDASYNC("g1p_eq_wrapper");
    NEGCMPCHECK(rows*512);
    NEGPRINTPASS(pass);

    // hext_fft -> hext -> h

    printf("hext_fft -> hext -> h\n"); fflush(stdout);

    CLOCKSTART;
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>((g1p_t *)g1p_tmp, (g1p_t *)g1p_tmp);
    CUDASYNC("g1p_ift_wrapper");
    fk20_hext2h<<<rows, 256>>>((g1p_t *)g1p_tmp);
    CLOCKEND;
    CUDASYNC("fk20_hext2h");

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, (g1p_t *)h);
    CUDASYNC("g1p_eq_wrapper");
    NEGCMPCHECK(rows*512);
    NEGPRINTPASS(pass);

    // h -> h_fft

    printf("h -> h_fft\n"); fflush(stdout);

    CLOCKSTART;
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>((g1p_t *)g1p_tmp, (g1p_t *)g1p_tmp);
    CUDASYNC("g1p_fft_wrapper");
    CLOCKEND;

    clearRes512;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, h_fft);
    CUDASYNC("g1p_eq_wrapper");
    NEGCMPCHECK(rows*512);
    NEGPRINTPASS(pass);
}

/*******************************************************************************

The testing functions follow an common template, described in ./doc/fk20test.md

*******************************************************************************/

/**
 * @brief Test for fr_fft: toeplitz_coefficients -> toeplitz_coefficients_fft
 *
 * @param rows number of blocks in the range [1,512]
 */
void toeplitz_coefficients2toeplitz_coefficients_fft_512(unsigned rows){
    PTRN_FRTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    printf("=== RUN   %s\n", "fr_fft: toeplitz_coefficients -> toeplitz_coefficients_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(fr_tmp, (fr_t *)toeplitz_coefficients);
        CUDASYNC("fr_fft_wrapper");
        CLOCKEND;

        clearRes;
        fr_eq_wrapper<<<256, 32>>>(cmp, rows*16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft);
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

/**
 * @brief Test for g1p_fft: h -> h_fft"
 *
 * @param rows number of blocks in the range [1,512]
 */
void h2h_fft_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);

    printf("=== RUN   %s\n", "g1p_fft: h -> h_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>((g1p_t *)g1p_tmp, h);
        CUDASYNC("g1p_fft_wrapper");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, h_fft);
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

/**
 * @brief Test for g1p_ift: h_fft -> h
 *
 * @param rows number of blocks in the range [1,512]
 */
void h_fft2h_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    printf("=== RUN   %s\n", "g1p_ift: h_fft -> h");

    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>((g1p_t *)g1p_tmp, h_fft);
        CUDASYNC("g1p_ift_wrapper");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<16, 32>>>(cmp, 512*512, (g1p_t *)g1p_tmp, h);
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

/**
 * @brief Test for g1p_ift: hext_fft -> h
 *
 * @param rows number of blocks in the range [1,512]
 */
void hext_fft2h_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    printf("=== RUN   %s\n", "g1p_ift: hext_fft -> h");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>((g1p_t *)g1p_tmp, hext_fft);
        CUDASYNC("g1p_ift_wrapper");
        fk20_hext2h<<<rows, 256>>>((g1p_t *)g1p_tmp);
        CUDASYNC("fk20_hext2h");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<8, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, h);
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

/**
 * @brief Test for fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients
 *
 * @param rows number of blocks in the range [1,512]
 */
void fk20_poly2toeplitz_coefficients_512(unsigned rows) {
    PTRN_FRTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_poly2toeplitz_coefficients<<<rows, 256 >>>(fr_tmp, polynomial);
        //IMPORTANT: This function does not need shared memory. Making the kernel call with a dynamic shared memory allocation
        //is known to cause some suble bugs, that not always show during normal execution.
        CUDASYNC("fk20_poly2toeplitz_coefficients");
        CLOCKEND;

        clearRes;
        fr_eq_wrapper<<<1, 32>>>(cmp, rows*16*512, fr_tmp, (fr_t *)toeplitz_coefficients);
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

/**
 * @brief Test for fk20_poly2hext_fft: polynomial -> hext_fft
 *
 * @param rows number of blocks in the range [1,512]
 */
void fk20_poly2hext_fft_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    pass = true;

    SET_SHAREDMEM(g1p_sharedmem, fk20_poly2hext_fft);

    printf("=== RUN   %s\n", "fk20_poly2hext_fft: polynomial -> hext_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_poly2hext_fft<<<rows, 256, fr_sharedmem>>>((g1p_t *)g1p_tmp, polynomial, (const g1p_t *)xext_fft);
        CUDASYNC("fk20_poly2hext_fft");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<1, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, (g1p_t *)hext_fft);
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

/**
 * @brief Test for fk20_poly2h_fft: polynomial -> h_fft
 *
 * @param rows number of blocks in the range [1,512]
 */
void fk20_poly2h_fft_512(unsigned rows){
    PTRN_G1PTMP; PTRN_FRTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    printf("=== RUN   %s\n", "fk20_poly2h_fft: polynomial -> h_fft");

    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_poly2h_fft((g1p_t *)g1p_tmp, polynomial, (const g1p_t *)xext_fft, rows);
        CUDASYNC("fk20_poly2h_fft");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<1, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, (g1p_t *)h_fft);
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

/**
 * @brief Test for hext_fft2h_fft_512: hext_fft -> h_fft
 *
 * @param rows number of blocks in the range [1,512]
 */
void hext_fft2h_fft_512_graph(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    printf("=== RUN   %s\n", "hext_fft2h_fft_512 graph: hext_fft -> h_fft");
    printf("=== WARN: rows set to 512 for this test");
    //Init by preparing the input
    rows = 512;
    g1p_t *input; 
    err = cudaMallocManaged(&input, 512*512*sizeof(g1p_t));
    if (err != cudaSuccess)
         printf("%s:%d  Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), "graph input");

    for (int i = 0; i < 512; i++) {
        for (int j = 0; j < 512; j++) {
            g1p_cpy(input[i * 512 + j], hext_fft[i * 512 + j]);
        }
    }
    
    CLOCKSTART;
    g1p512SquareTranspose(input);
    end = clock(); 
    printf(" (%.1f ms transpose overhead)\n", (end - start) * (1000. / ((__clock_t) 1000000)));

    for(int testIDX=0; testIDX<=1; testIDX++){
        CLOCKSTART;
        fk20_hext_fft_2_h_fft_512((g1p_t *)g1p_tmp, input);
        CUDASYNC("fk20_hext_fft_2_h_fft_512");
        CLOCKEND;
        
        //transpose output
        g1p512SquareTranspose((g1p_t *)g1p_tmp);

        clearRes;
        g1p_eq_wrapper<<<8, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, h_fft);
        CUDASYNC("g1p_eq_wrapper");

        if (testIDX == 0){
            CMPCHECK(rows*512)
            PRINTPASS(pass);
            }
        else{
            NEGCMPCHECK(rows*512);
            NEGPRINTPASS(pass);
        }

        varMangle(input, 512*512, 32);
    }
}

/**
 * @brief Test for hext_fft2h_fft_512: hext_fft -> h_fft
 * Using CUDA graph
 *
 * @param rows number of blocks in the range [1,512]
 */
void hext_fft2h_fft_512(unsigned rows){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    CLOCKINIT;

    printf("=== RUN   %s\n", "hext_fft2h_fft_512 graph: hext_fft -> h_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_hext_fft2h_fft<<<rows, 256, g1p_sharedmem>>>((g1p_t *)g1p_tmp, hext_fft);
        CUDASYNC("fk20_hext_fft2h_fft");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<8, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, h);
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

/**
 * @brief Test for fk20_msm: Toeplitz_coefficients+xext_fft -> hext_fft
 *
 * @param rows number of blocks in the range [1,512]
 */
void fk20_msmloop_512(unsigned rows){
    CLOCKINIT;
    cudaError_t err;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_msm: Toeplitz_coefficients+xext_fft -> hext_fft");
    for(int testIDX=0; testIDX<=1; testIDX++){

        CLOCKSTART;
        fk20_msm<<<rows, 256>>>((g1p_t *)g1p_tmp, (const fr_t*)toeplitz_coefficients_fft, (const g1p_t*)xext_fft);
        CUDASYNC("fk20_msm");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, (g1p_t *)hext_fft);
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

static bool lut_compute = false;
/**
 * @brief Test for fk20_msm: Toeplitz_coefficients+xext_fft -> hext_fft
 *
 * @param rows number of blocks in the range [1,512]
 */
void fk20_msmcomb_512(unsigned rows){
    CLOCKINIT;
    cudaError_t err;
    bool pass = true;

    rows = 512;
    if(!lut_compute){
        printf("=== INFO  %s\n", "Computing LUT");
        fk20_msm_makelut<<<dim3(512, 16, 1), 1>>>(xext_lut, (const g1p_t (*)[512])(xext_fft));
        CUDASYNC("fk20_msm_makelut");
        lut_compute = true;
    }

    printf("=== RUN   %s\n", "fk20_msm_comb: Toeplitz_coefficients+xext_fft -> hext_fft");
    printf("=== INFO  %s\n", "Overriding rows to 512 for this test");
    for(int testIDX=0; testIDX<=1; testIDX++){
        CLOCKSTART;
        fk20_msm_comb<<<512, 256>>>(g1p_tmp, (const fr_t (*)[16][512])(toeplitz_coefficients_fft), xext_lut);
        CUDASYNC("fk20_msm_comb");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, (g1p_t *)hext_fft);
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

/**
 * @brief Test for fk20_msm: Toeplitz_coefficients+xext_fft -> hext_fft
 *
 * @param rows number of blocks in the range [1,512]
 */
void fk20_msmcomb_512_graph(unsigned rows){
    CLOCKINIT;
    cudaError_t err;
    bool pass = true;

    rows = 512;
    if(!lut_compute){
        printf("=== INFO  %s\n", "Computing LUT");
        fk20_msm_makelut<<<dim3(512, 16, 1), 1>>>(xext_lut, (const g1p_t (*)[512])(xext_fft));
        CUDASYNC("fk20_msm_makelut");
        lut_compute = true;
    }

    printf("=== RUN   %s\n", "fk20_msm_comb: Toeplitz_coefficients+xext_fft -> hext_fft");
    printf("=== INFO  %s\n", "Overriding rows to 512 for this test");
    for(int testIDX=0; testIDX<=1; testIDX++){
        CLOCKSTART;
        fk20_msm_comb_graph(g1p_tmp, (const fr_t (*)[16][512])(toeplitz_coefficients_fft), xext_lut);
        CUDASYNC("fk20_msm_comb");
        CLOCKEND;

        clearRes;
        g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, (g1p_t *)g1p_tmp, (g1p_t *)hext_fft);
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


//Deprecated funtion
#if 0
    void fk20_poly2toeplitz_coefficients_fft_test(unsigned rows){
        // Test for deprecated function.
        PTRN_FRTMP;
        CLOCKINIT;
        cudaError_t err;
        bool pass = true;

        printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients_fft: polynomial -> toeplitz_coefficients_fft");
        memset(fr_tmp, 0xdeadbeef,512*16*512*sizeof(fr_t)); //pattern on tmp dest.
        CLOCKSTART;
        fk20_poly2toeplitz_coefficients_fft<<<rows, 256>>>(fr_tmp, polynomial);
        err = cudaDeviceSynchronize();
        CUDASYNC("fk20_poly2toeplitz_coefficients_fft");
        CLOCKEND;
        clearRes;
        fr_eq_wrapper<<<16, 256>>>(cmp, rows*16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft);
        CUDASYNC("fr_eq_wrapper");
        // Check result

        CMPCHECK(rows*16*512);
        PRINTPASS(pass);
    }
#endif

/**
 * @brief initializes memory and load KAT
 * 
 */
void katInit(){
    #define MALLOCSYNC(fmt, ...) \
    if (err != cudaSuccess)                                                                                            \
    printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)

    const char* FILENAME = "fk20_512_kat.bin";
    cudaError_t err;

    err = cudaMallocManaged(&xext_fft,  16*512 * sizeof(g1p_t)); 
        MALLOCSYNC("xext_fft");
    err = cudaMallocManaged(&polynomial,  512*4096 * sizeof(fr_t)); 
        MALLOCSYNC("polynomial");
    err = cudaMallocManaged(&toeplitz_coefficients,  512*16*512 * sizeof(fr_t)); 
        MALLOCSYNC("toeplitz_coefficients");
    err = cudaMallocManaged(&toeplitz_coefficients_fft,  512*16*512 * sizeof(fr_t)); 
        MALLOCSYNC("toeplitz_coefficients_fft");
    err = cudaMallocManaged(&hext_fft,  512*512 * sizeof(g1p_t)); 
        MALLOCSYNC("hext_fft");
    err = cudaMallocManaged(&h,  512*512 * sizeof(g1p_t)); 
        MALLOCSYNC("h");
    err = cudaMallocManaged(&h_fft,  512*512 * sizeof(g1p_t)); 
        MALLOCSYNC("h_fft");

    size_t bytesRead=0;
    size_t bytesExpected = 449970176;

    if(access(FILENAME, F_OK) != 0){
        printf("ERROR: File %s does not exist.\n", FILENAME);
        printf("Try running fk20_512test_boostrap.\n");
        exit(-1);
    }

    FILE *file = fopen(FILENAME, "r");

    if (file == NULL){
        printf("ERROR: Failed to open %s.\n", FILENAME);
        exit(-1);

    }

    bytesRead += fread(xext_fft                 , 1,  16*512 * sizeof(g1p_t), file);
    bytesRead += fread(polynomial               , 1,  512*4096 * sizeof(fr_t), file);
    bytesRead += fread(toeplitz_coefficients    , 1,  512*16*512 * sizeof(fr_t), file);
    bytesRead += fread(toeplitz_coefficients_fft, 1,  512*16*512 * sizeof(fr_t), file);
    bytesRead += fread(hext_fft                 , 1,  512*512 * sizeof(g1p_t), file);
    bytesRead += fread(h                        , 1,  512*512 * sizeof(g1p_t), file);
    bytesRead += fread(h_fft                    , 1,  512*512 * sizeof(g1p_t), file);

    if (bytesRead!=bytesExpected){
        printf("ERROR: mismatch in read size: %lu, exp: %lu\n", bytesRead, bytesExpected);
        printf("Try running fk20_512test_boostrap again.\n");
        exit(-1);
    }

}


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//           Useful functions for the falsifiability tests                    //
//                                                                            //
// Useful for the Falsifiability tests                                        //
// If you are using a variable where i*step == i*step+1, you can end up with  //
// a false(false positive).                                                   //
// A staggered start helps to mitigate it, but it can happen with a very      //
// small probability.                                                         //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#define START_INDEX 3

/**
 * @brief swap elements at positions multiple of step. Nondestructive, call
 * a second time to undo the changes
 *
 * @param[out] target Pointer to array
 * @param[in] size length of the array
 * @param[in] step distance between elements swapped.
 */
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

/**
 * @brief swap elements at positions multiple of step. Nondestructive, call
 * a second time to undo the changes
 *
 * @param[out] target Pointer to array
 * @param[in] size length of the array
 * @param[in] step distance between elements swapped.
 */
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

#undef START_INDEX
// vim: ts=4 et sw=4 si
