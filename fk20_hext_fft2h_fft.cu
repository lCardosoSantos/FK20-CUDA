// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

/**
 * @brief hext_fft -> h_fft
 * 
 * Grid must be 1-D, 256 threads per block.
 * Dynamic shared memory: g1p_sharedmem(73728 Bytes)
 * 
 * @param[out] h_fft array with dimensions [gridDim.x * 512]
 * @param[in] hext_fft array with dimensions [gridDim.x * 512]
 * @return void 
 */
__global__ void fk20_hext_fft2h_fft(g1p_t *h_fft, const g1p_t *hext_fft){
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    hext_fft += 512*bid;
    h_fft    += 512*bid;

    // hext_fft -> h -> h_fft

    // h = ift hext_fft
    g1p_ift(h_fft, hext_fft);
    __syncthreads();

    // zero second half of h
    g1p_inf(h_fft[256+tid]);
    __syncthreads();

    // h_fft = fft h
    g1p_fft(h_fft, h_fft);
}

// vim: ts=4 et sw=4 si
