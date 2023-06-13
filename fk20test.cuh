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
void fk20_poly2toeplitz_coefficients_test();
void fk20_poly2hext_fft_test();
void fk20_poly2h_fft_test();

void toeplitz_coefficients2toeplitz_coefficients_fft();
void h2h_fft();
void h_fft2h();
void hext_fft2h();

//pretty print
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_RESET   "\x1b[0m"
#define COLOR_BOLD    "\x1b[1m"

#define PRINTPASS(pass) printf("--- %s\n", pass ? COLOR_GREEN "PASS" COLOR_RESET: COLOR_RED COLOR_BOLD "FAIL" COLOR_RESET);

#endif // FK20_TEST_CUH

// vim: ts=4 et sw=4 si
