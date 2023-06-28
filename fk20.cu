#include <stdio.h>

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

// Workspace in shared memory

extern __shared__ fr_t fr_tmp[];    // 16 KiB shared memory
extern __shared__ g1p_t g1p_tmp[];  // 72 KiB shared memory

////////////////////////////////////////////////////////////////////////////////

// fk20_setup2xext_fft(): setup[0] -> xext_fft

__global__ void fk20_setup2xext_fft(g1p_t *xext_fft, const g1p_t *setup) {

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

// fk20_hext_fft2hext(): hext_fft -> hext

// parameters:
// - in  hext_fft   array with 512*gridDim.x elements
// - out hext       array with 512*gridDim.x elements

__global__ void fk20_hext_fft2hext(g1p_t *hext, const g1p_t *hext_fft) {
    g1p_ift(hext, hext_fft);
}

////////////////////////////////////////////////////////////////////////////////

// fk20_h2h_fft(): h -> h_fft

// parameters:
// - in  h      array with 512*gridDim.x elements
// - out h_fft  array with 512*gridDim.x elements

__global__ void fk20_h2h_fft(g1p_t *h_fft, const g1p_t *h) {
    g1p_fft(h_fft, h);
}

// vim: ts=4 et sw=4 si


    //debug, copy g1p_tmp to output
//    g1p_cpy(h_fft[tid+  0], g1p_tmp[tid+  0]);
//    g1p_cpy(h_fft[tid+256], g1p_tmp[tid+256]);
