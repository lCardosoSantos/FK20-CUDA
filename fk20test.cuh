// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FK20_TEST_CUH
#define FK20_TEST_CUH
#include <stdio.h>
// Shared memory sizes

const size_t g1p_sharedmem = 512*3*6*8; // 512 points * 3 residues/point * 6 words/residue * 8 bytes/word = 72 KiB
const size_t fr_sharedmem = 512*4*8; // 512 residues * 4 words/residue * 8 bytes/word = 16 KiB

// Complete tests

void FK20TestFFT();
void FK20TestPoly();
void FK20TestFFTRand(FILE *inputStream);

//Single tests
void fk20_poly2toeplitz_coefficients_test(fr_t polynomial[4096], fr_t toeplitz_coefficients[16][512]);
void fk20_poly2hext_fft_test(fr_t polynomial_l[4096], g1p_t xext_fft_l[16][512], g1p_t hext_fft_l[512]);
void fk20_poly2h_fft_test(fr_t polynomial_l[4096], g1p_t xext_fft_l[16][512], g1p_t h_fft_l[512]);

void toeplitz_coefficients2toeplitz_coefficients_fft(fr_t toeplitz_coefficients_l[16][512], fr_t toeplitz_coefficients_fft_l[16][512]);
void h2h_fft(g1p_t h_l[512], g1p_t h_fft_l[512]);
void h_fft2h(g1p_t h_fft_l[512], g1p_t h_l[512]);
void hext_fft2h(g1p_t hext_fft_l[512], g1p_t h_l[512]);

//pretty print
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_RESET   "\x1b[0m"
#define COLOR_BOLD    "\x1b[1m"

#define PRINTPASS(pass) printf("--- %s\n", pass ? COLOR_GREEN "PASS" COLOR_RESET: COLOR_RED COLOR_BOLD "FAIL" COLOR_RESET);

//debug macros for dumping elements to file
#define WRITEU64(writing_stream, var, nu64Elem) do{ \
    uint64_t *pointer = (uint64_t *)(*var); \
    for (int count=0; count<(nu64Elem); count++){ \
        fprintf(writing_stream,"%016lx\n",pointer[count]); \
    } \
}while(0)

#define WRITEU64TOFILE(fileName, var, nu64Elem) do{ \
    FILE * filepointer = fopen(fileName, "w");     \
    WRITEU64(filepointer, var, (nu64Elem));           \
    fclose(filepointer);                           \
}while(0) 

//sadly cuda doesn't allow fprintf inside a kernel, so printfItis.
#define WRITEU64STDOUT(var, nu64Elem) do{ \
    uint64_t *pointer = (uint64_t *)(*var); \
    for (int count=0; count<(nu64Elem); count++){ \
        printf("%016lx\n",pointer[count]); \
    } \
}while(0)


#endif // FK20_TEST_CUH

// vim: ts=4 et sw=4 si
