// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FK20_TEST_CUH
#define FK20_TEST_CUH

#include <stdio.h>

#include "test.inc"

// Shared memory sizes

const size_t g1p_sharedmem = 512 * 3 * 6 * 8; // 512 points * 3 residues/point * 6 words/residue * 8 bytes/word = 72 KiB
const size_t fr_sharedmem = 512 * 4 * 8;      // 512 residues * 4 words/residue * 8 bytes/word = 16 KiB

// Complete tests

void FK20TestFFT();
void FK20TestPoly();
void FK20TestFFTRand(FILE *inputStream);

// Single tests
void fk20_poly2toeplitz_coefficients_test(fr_t polynomial[4096], fr_t toeplitz_coefficients[16][512]);
// void fk20_poly2toeplitz_coefficients_fft_test(fr_t polynomial_l[4096], fr_t toeplitz_coefficients__fftl[16][512]); //deprecated
void fk20_poly2hext_fft_test(fr_t polynomial_l[4096], g1p_t xext_fft_l[16][512], g1p_t hext_fft_l[512]);
void fk20_poly2h_fft_test(fr_t polynomial_l[4096], g1p_t xext_fft_l[16][512], g1p_t h_fft_l[512]);
void fk20_msmloop(g1p_t hext_fft_l[512], fr_t toeplitz_coefficients_fft_l[16][512], g1p_t xext_fft_l[16][512]);

void toeplitz_coefficients2toeplitz_coefficients_fft(fr_t toeplitz_coefficients_l[16][512],
                                                     fr_t toeplitz_coefficients_fft_l[16][512]);
void h2h_fft(g1p_t h_l[512], g1p_t h_fft_l[512]);
void h_fft2h(g1p_t h_fft_l[512], g1p_t h_l[512]);
void hext_fft2h(g1p_t hext_fft_l[512], g1p_t h_l[512]);
void hext_fft2h_fft(g1p_t hext_fft_l[512], g1p_t h_fft_l[512]);

void fullTest();

#endif // FK20_TEST_CUH

// vim: ts=4 et sw=4 si
