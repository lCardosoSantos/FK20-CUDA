// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

/**
 * This standalone executable uses the older method of compiling and linking the fk20_512 tests, and writes them to a blob.
 * Only use when different test values are needed.
 * 
 */
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include "g1.cuh"

//Original vars
extern __managed__ g1p_t xext_fft[16][512];
extern __managed__ fr_t polynomial[512*4096];
// Intermediate values
extern __managed__ fr_t toeplitz_coefficients[512*16][512];
extern __managed__ fr_t toeplitz_coefficients_fft[512][16][512];
extern __managed__ g1p_t hext_fft[512*512];
extern __managed__ g1p_t h[512*512];
// Test vector output
extern __managed__ g1p_t h_fft[512*512];

const char* FILENAME = "fk20_512_kat.bin";


void write(){
    if(access(FILENAME, F_OK) == 0){
        printf("ERROR: File %s already exists.\n", FILENAME);
        exit(EEXIST);
    }

    FILE *file = fopen(FILENAME, "w");

    if (file == NULL){
        printf("ERROR: Failed to create %s.\n", FILENAME);
        exit(-1);

    }

    size_t bytesWritten = 0;
    size_t bytesExpected = 
        16  * 512  *       sizeof(g1p_t)+ \
        512 * 4096 *       sizeof(fr_t) + \
        512 * 16   * 512 * sizeof(fr_t) + \
        512 * 16   * 512 * sizeof(fr_t) + \
        512 * 512  *       sizeof(g1p_t)+ \
        512 * 512  *       sizeof(g1p_t)+ \
        512 * 512  *       sizeof(g1p_t);

    bytesWritten += fwrite(xext_fft                 , 1,  16*512 * sizeof(g1p_t), file);
    bytesWritten += fwrite(polynomial               , 1,  512*4096 * sizeof(fr_t), file);
    bytesWritten += fwrite(toeplitz_coefficients    , 1,  512*16*512 * sizeof(fr_t), file);
    bytesWritten += fwrite(toeplitz_coefficients_fft, 1,  512*16*512 * sizeof(fr_t), file);
    bytesWritten += fwrite(hext_fft                 , 1,  512*512 * sizeof(g1p_t), file);
    bytesWritten += fwrite(h                        , 1,  512*512 * sizeof(g1p_t), file);
    bytesWritten += fwrite(h_fft                    , 1,  512*512 * sizeof(g1p_t), file);

    if (bytesWritten!=bytesExpected){
        printf("ERROR: mismatch in write size: %lu, exp: %lu\n", bytesWritten, bytesExpected);
    }
    else{
        printf("%lu bytes written to %s\n", bytesWritten, FILENAME);
    }

    fclose(file);
}

#define MALLOCSYNC(fmt, ...) \
    if (err != cudaSuccess)                                                                                            \
    printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)

void test(){
    //Allocate local vars
    g1p_t *l_xext_fft;
    fr_t  *l_polynomial;
    fr_t  *l_toeplitz_coefficients;
    fr_t  *l_toeplitz_coefficients_fft;
    g1p_t *l_hext_fft;
    g1p_t *l_h;
    g1p_t *l_h_fft;

    cudaError_t err;

    err = cudaMallocManaged(&l_xext_fft,  16*512 * sizeof(g1p_t)); 
        MALLOCSYNC("xext_fft");
    err = cudaMallocManaged(&l_polynomial,  512*4096 * sizeof(fr_t)); 
        MALLOCSYNC("polynomial");
    err = cudaMallocManaged(&l_toeplitz_coefficients,  512*16*512 * sizeof(fr_t)); 
        MALLOCSYNC("toeplitz_coefficients");
    err = cudaMallocManaged(&l_toeplitz_coefficients_fft,  512*16*512 * sizeof(fr_t)); 
        MALLOCSYNC("toeplitz_coefficients_fft");
    err = cudaMallocManaged(&l_hext_fft,  512*512 * sizeof(g1p_t)); 
        MALLOCSYNC("hext_fft");
    err = cudaMallocManaged(&l_h,  512*512 * sizeof(g1p_t)); 
        MALLOCSYNC("h");
    err = cudaMallocManaged(&l_h_fft,  512*512 * sizeof(g1p_t)); 
        MALLOCSYNC("h_fft");

    //Read into local vars
    size_t bytesRead=0;
    size_t bytesExpected = 
    16  * 512  *       sizeof(g1p_t)+ \
    512 * 4096 *       sizeof(fr_t) + \
    512 * 16   * 512 * sizeof(fr_t) + \
    512 * 16   * 512 * sizeof(fr_t) + \
    512 * 512  *       sizeof(g1p_t)+ \
    512 * 512  *       sizeof(g1p_t)+ \
    512 * 512  *       sizeof(g1p_t); //449970176

    if(access(FILENAME, F_OK) != 0){
        printf("ERROR: File %s does not exist.\n", FILENAME);
        exit(EEXIST);
    }

    FILE *file = fopen(FILENAME, "r");

    if (file == NULL){
        printf("ERROR: Failed to open %s.\n", FILENAME);
        exit(-1);

    }

    bytesRead += fread(l_xext_fft                 , 1,  16*512 * sizeof(g1p_t), file);
    bytesRead += fread(l_polynomial               , 1,  512*4096 * sizeof(fr_t), file);
    bytesRead += fread(l_toeplitz_coefficients    , 1,  512*16*512 * sizeof(fr_t), file);
    bytesRead += fread(l_toeplitz_coefficients_fft, 1,  512*16*512 * sizeof(fr_t), file);
    bytesRead += fread(l_hext_fft                 , 1,  512*512 * sizeof(g1p_t), file);
    bytesRead += fread(l_h                        , 1,  512*512 * sizeof(g1p_t), file);
    bytesRead += fread(l_h_fft                    , 1,  512*512 * sizeof(g1p_t), file);

    if (bytesRead!=bytesExpected){
        printf("ERROR: mismatch in read size: %lu, exp: %lu\n", bytesRead, bytesExpected);
        exit(-1);
    }
    else{
        printf("%lu bytes read from %s\n", bytesRead, FILENAME);
    }

    //deep check if the read was sucessfull
    #define CHECK(var, size)                \
    if(memcmp(var, l_##var, size) != 0){    \
        cErr = 1;    \
        printf("Error: " #var " fails!\n"); \
    }

    int cErr = 0;
    CHECK(xext_fft                 , 16*512 * sizeof(g1p_t));
    CHECK(polynomial               , 512*4096 * sizeof(fr_t));
    CHECK(toeplitz_coefficients    , 512*16*512 * sizeof(fr_t));
    CHECK(toeplitz_coefficients_fft, 512*16*512 * sizeof(fr_t));
    CHECK(hext_fft                 , 512*512 * sizeof(g1p_t));
    CHECK(h                        , 512*512 * sizeof(g1p_t));
    CHECK(h_fft                    , 512*512 * sizeof(g1p_t));

    if (!cErr){
        printf("Load check failed!\n");
    }

}


int main(int argc, char const *argv[])
{
    write();
    test();

    return 0;
}

