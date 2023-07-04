// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FK20_CUH
#define FK20_CUH

#include <stdint.h>

#include "fr.cuh"
#include "g1.cuh"

// External interface

__global__ void fk20_setup2xext_fft(g1p_t xext_fft[8192], const g1p_t *setup);

__global__ void fk20_poly2toeplitz_coefficients(fr_t *toeplitz_coefficients, const fr_t *polynomial);
__global__ void fk20_poly2toeplitz_coefficients_fft(fr_t *toeplitz_coefficients_fft, const fr_t *polynomial);
__global__ void fk20_poly2hext_fft(g1p_t *hext_fft, const fr_t *polynomial, const g1p_t xext_fft[8192]);
__global__ void fk20_poly2h_fft(g1p_t *h_fft, const fr_t *polynomial, const g1p_t xext_fft[8192]);

__global__ void fk20_hext_fft2h(g1p_t *h, const g1p_t *hext_fft);
__global__ void fk20_h2h_fft(g1p_t *h_fft, const g1p_t *h);
__global__ void fk20_hext_fft2h_fft(g1p_t *h_fft, const g1p_t *hext_fft);
__global__ void fk20_msm_xext_fftANDtoepliz_fft2hext_fft(g1p_t *hext_fft_l, const fr_t *toeplitz_coefficients_fft_l, const g1p_t *xext_fft);

#endif

// vim: ts=4 et sw=4 si
