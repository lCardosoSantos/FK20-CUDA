// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

// Workspace in shared memory

//extern __shared__ fr_t fr_tmp[];    // 16 KiB shared memory
//extern __shared__ g1p_t g1p_tmp[];  // 72 KiB shared memory

////////////////////////////////////////////////////////////////////////////////

/**
 * @brief setup -> xext_fft
 *
 * Grid must be 16, 256 threads per block.
 *
 * @param[out] xext_fft array with dimension [16*512]
 * @param setup array with dimension [16*512]
 * @return void
 */
__global__ void fk20_setup2xext_fft(g1p_t *xext_fft, const g1p_t *setup) {
    //TODO: Not passing test, probably bad block indexing
    if (gridDim.x  !=  16) return;
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    const int n = 4096, l = 16, k = 256;

    g1p_t *xext = xext_fft;

    int input = n - 1 - bid - l * tid;
    int output = 2*k * bid + tid;

    if (input >= 0)
        g1p_cpy(xext[output], setup[input]);
    else
        g1p_inf(xext[output]);

    // Part 1: extend with point at infinity, then perform G1 FFT

    __syncthreads();

    g1p_inf(xext[2*k*bid + k + tid]);

    g1p_fft(xext_fft, xext);  // 16 FFT-512
}



////////////////////////////////////////////////////////////////////////////////
// These functions are syntax sugar.
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief hext_fft -> hext
 *
 * @param[in] hext array with 512*gridDim.x elements
 * @param[out] hext_fft array with 512*gridDim.x elements
 * @return
 */
__global__ void fk20_hext_fft2hext(g1p_t *hext, const g1p_t *hext_fft) {
    g1p_ift(hext, hext_fft);
}

////////////////////////////////////////////////////////////////////////////////

// fk20_h2h_fft(): h -> h_fft

// parameters:
// - in  h      array with 512*gridDim.x elements
// - out h_fft  array with 512*gridDim.x elements

/**
 * @brief h -> h_fft
 *
 * @param[out] h_fft array with 512*gridDim.x elements
 * @param[in] h array with 512*gridDim.x elements
 * @return void
 */
__global__ void fk20_h2h_fft(g1p_t *h_fft, const g1p_t *h) {
    g1p_fft(h_fft, h);
}

// vim: ts=4 et sw=4 si
