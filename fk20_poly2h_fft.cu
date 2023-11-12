// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <time.h>
#include <stdio.h> // For reporting sharedmem errors.

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

// Maximum number of rows covered in the test.
#define ROWS 512

static __managed__ fr_t   fr[ROWS * 16 * 512]; // 16 KiB per threadblock
static __managed__ g1p_t g1p[ROWS * 512];      // 72 KiB per threadblock

////////////////////////////////////////////////////////////////////////////////

/**
 * @brief hext -> h
 * Fill upper half of hext with inf, modifying in place.
 *
 * @param[in,out] h
 * @return void
 */
__global__ void fk20_hext2h(g1p_t *h) {
    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    h += 512 * bid;
    g1p_inf(h[256 + tid]);
}

/**
 * @brief polynomial + xext_fft -> h_fft
 * This function is a wrapper for the full FK20 computation, i.e. commits up to 512
 * polynomials of 4096 elements to the same setup.
 * l = 16, intrinsic to the implementation.
 *
 * @param[out] h_fft        array with dimensions [rows * 512]
 * @param[in]  polynomial   array with dimensions [rows * 16 * 512]
 * @param[in]  xext_fft     array with dimensions [16 * 512]
 * @param[in]  rows number of rows (gridDim.x)
 * @return void
 */
__host__ void fk20_poly2h_fft(g1p_t *h_fft, const fr_t *polynomial, const g1p_t xext_fft[8192], unsigned rows) {
    cudaError_t err;

    // Setup

    SET_SHAREDMEM(fr_sharedmem, fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    // polynomial -> tc
    fk20_poly2toeplitz_coefficients<<<rows, 256, fr_sharedmem>>>(fr, polynomial);
    // CUDASYNC("fk20_poly2toeplitz_coefficients");

    // tc -> tc_fft
    fr_fft_wrapper<<<rows * 16, 256, fr_sharedmem>>>(fr, fr);
    // CUDASYNC("fr_fft_wrapper");

    // tc_fft -> hext_fft
    fk20_msm<<<rows, 256>>>(g1p, fr, xext_fft);
    // CUDASYNC("fk20_msm");

    // hext_fft -> hext
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p, g1p);
    // CUDASYNC("g1p_ift_wrapper");

    // hext -> h
    fk20_hext2h<<<rows, 256>>>(g1p);
    // CUDASYNC("fk20_hext2h");

    // h -> h_fft
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(h_fft, g1p);
    // CUDASYNC("g1p_fft_wrapper");
}
// vim: ts=4 et sw=4 si
