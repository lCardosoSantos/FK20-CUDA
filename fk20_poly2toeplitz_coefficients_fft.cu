// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fr.cuh"
#include "fk20.cuh"

static __device__ fr_t fr_tmp[512*16*512];     // 256 KiB memory per threadblock

/**
 * @brief  polynomial -> toeplitz_coefficients_fft
 *
 * WARN: Usage of this function is deprecated: Instead use fk20_poly2toeplitz_coefficients()
 * followed by fr_fft(). This function is not covered in the test suite.
 * This function remains in the repository for future optimizations.
 *
 * @param[out] toeplitz_coefficients_fft array with 16*512*gridDim.x elements
 * @param[in] polynomial array with 16*512*gridDim.x elements
 * @return void
 */
__global__ void fk20_poly2toeplitz_coefficients_fft(fr_t *toeplitz_coefficients_fft, const fr_t *polynomial) {

    // gridDim.x is the number of rows
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;  // k
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    // Accumulators and temporaries in registers or local
    // (thread-interleaved global) memory

    polynomial += 4096 * bid;
    toeplitz_coefficients_fft += 8192 * bid;

    fr_t *fr = fr_tmp + 8192 * bid;

    __syncthreads();

    // Loop

    for (int i=0; i<16; i++) {

        // Copy from the polynomial into half of the coefficient array

        unsigned src = tid*16+15-i;
        unsigned dst = (tid+257)%512;

        if (tid > 0)
            fr_cpy(fr[dst], polynomial[src]);
        else
            fr_zero(fr[dst]);

        __syncthreads();

        // Zero the other half of coefficients before FFT

        fr_zero(fr[tid+1]);

        __syncthreads();

        // Compute FFT

        fr_fft(fr, fr);

        __syncthreads();

        fr_cpy(toeplitz_coefficients_fft[tid], fr[tid]);
        fr_cpy(toeplitz_coefficients_fft[tid+256], fr[tid+256]);

        toeplitz_coefficients_fft += 512;
    }
}

// vim: ts=4 et sw=4 si
