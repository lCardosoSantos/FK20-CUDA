// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

/**
 * @brief polynomial -> toeplitz_coefficients
 *
 * @param[out] toeplitz_coefficients  array with dimension [4096 * gridDim.x]
 * @param[in] polynomial array with dimensions [rows * 16 * 512]
 * @return void
 *
 * Grid must be 1-D, 256 threads per block.
 *
 * IMPORTANT: This function does not need shared memory. Making the kernel call with a dynamic shared memory allocation
 * is known to cause some subtle bugs, that not always show during normal execution.
 * Similar comment is present in fk20test_poly.cu and fk20_512test_poly.cu. In case this function changes and starts
 * needing shared memory, correct the tests on those two files.
 */
__global__ void fk20_poly2toeplitz_coefficients(fr_t *toeplitz_coefficients, const fr_t *polynomial) {

    // gridDim.x is the number of rows
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    polynomial += 4096 * bid;
    toeplitz_coefficients += 8192 * bid;

    for (int i=0; i<16; i++) {

        // Copy from the polynomial into half of the coefficient array

        unsigned src = tid*16+15-i;
        unsigned dst = (tid+257)%512 + 512*i;

        if (tid > 0)
            fr_cpy(toeplitz_coefficients[dst], polynomial[src]);
        else
            fr_zero(toeplitz_coefficients[dst]);

        __syncwarp(0xffffffff);

        // Zero the other half of coefficients before FFT

        fr_zero(toeplitz_coefficients[512*i+tid+1]);
    }
}

// vim: ts=4 et sw=4 si
