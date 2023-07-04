#include <stdio.h>

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

////////////////////////////////////////////////////////////////////////////////
// fk20_hext_fft2h_fft(): h_fft -> hext_fft
//
// parameters:
// -in hext_fft    g1p_t hext_fft[blockDim.x][512]
// -out h_fft      g1p_T h_fft[blockDim.x][512]
//
__global__ void fk20_hext_fft2h_fft(g1p_t *h_fft, const g1p_t *hext_fft){
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    g1p_t * hextfft = (g1p_t *)hext_fft+512*bid;
    h_fft += 512*bid;


    // hext_fft -> h -> h_fft
    // h = ift hext_fft
    __syncthreads();
    g1p_ift(h_fft, hext_fft);
    __syncthreads();
    // zero second half of h
    g1p_inf(h_fft[256+tid]);
    // h_fft = fft h
    __syncthreads();
    g1p_fft(h_fft, h_fft);

}