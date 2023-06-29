#include <stdio.h>

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"
// Workspace in shared memory

//extern __shared__ fr_t fr_tmp[];    // 16 KiB shared memory
//extern __shared__ g1p_t g1p_tmp[];  // 72 KiB shared memory

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

////////////////////////////////////////////////////////////////////////////////

// fk20_poly2toeplitz_coefficients(): polynomial -> toeplitz_coefficients

// parameters:
// - in  polynomial array with 4096*gridDim.x elements
// - out toeplitz_coefficients array with 8192*gridDim.x elements

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

        __syncthreads();

        // Zero the other half of coefficients before FFT

        fr_zero(toeplitz_coefficients[512*i+tid+1]);
    }
    
}

////////////////////////////////////////////////////////////////////////////////

// fk20_poly2hext_fft(): polynomial + xext_fft -> hext_fft

// parameters:
// - in  polynomial array with 4096*gridDim.x elements
// - in  xext_fft   array with 16*512 elements, computed using fk20_setup2xext_fft()
// - out hext_fft   array with    512*gridDim.x elements

// Note: shared memory is used both in MSM loop and FFTs, without conflict

__global__ void fk20_poly2hext_fft(g1p_t *hext_fft, const fr_t *polynomial, const g1p_t xext_fft[8192]) {
    extern __shared__ int sharedmem[];
    fr_t *frtmp = (fr_t *)(sharedmem);
    
    // gridDim.x is the number of rows
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    // Accumulators and temporaries in registers or local
    // (thread-interleaved global) memory

    g1p_t a0, a1, t;

    polynomial += 4096 * bid;
    hext_fft += 512 * bid;

    // MSM Loop

    for (int i=0; i<16; i++) {

        // Copy from the polynomial into half of the coefficient array

        unsigned src = tid*16+15-i;
        unsigned dst = (tid+257)%512;

        if (tid > 0)
            fr_cpy(frtmp[dst], polynomial[src]);
        else
            fr_zero(frtmp[dst]);

        __syncthreads();

        // Zero the other half of coefficients before FFT

        fr_zero(frtmp[tid+1]);

        // Compute FFT

        __syncthreads();
        fr_fft(frtmp, frtmp);
        __syncthreads();

        // multiply and accumulate

        g1p_cpy(t, xext_fft[512*i+tid+0]);
        g1p_mul(t, frtmp[tid]);
        __syncthreads();
        g1p_add(a0, t);

        g1p_cpy(t, xext_fft[512*i+tid+256]);
        g1p_mul(t, frtmp[tid+256]);
        __syncthreads();
        g1p_add(a1, t);
    }

    // Store accumulators

    g1p_cpy(hext_fft[tid+  0], a0);
    g1p_cpy(hext_fft[tid+256], a1);
}

////////////////////////////////////////////////////////////////////////////////

// fk20_poly2h_fft(): polynomial + xext_fft -> h_fft

// parameters:
// - in  xext_fft   array with 16*512 elements, computed using fk20_setup2xext_fft()
// - in  polynomial array with 16*512*gridDim.x elements
// - out h_fft      array with    512*gridDim.x elements

// Note: shared memory is used both in MSM loop and FFTs, without conflict
__global__ void fk20_poly2h_fft(g1p_t *h_fft, const fr_t *polynomial, const g1p_t xext_fft[8192]) {
    extern __shared__ int sharedmem[];
    fr_t *frtmp = (fr_t *)(sharedmem);
    g1p_t *g1ptmp = (g1p_t *)(sharedmem);

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

    //g1p_t a0={0}; 
    //g1p_t a1={0}; 
    //g1p_t t={0};
    
    //TODO: this is the correct way but breaks
    g1p_t a0, a1, t; 
    g1p_inf(a0);
    g1p_inf(a1);


    polynomial += 4096 * bid;
    h_fft += 512 * bid;

     // MSM Loop

    for (int i=0; i<16; i++) {

        // Copy from the polynomial into half of the coefficient array

        unsigned src = tid*16+15-i;
        unsigned dst = (tid+257)%512;

        if (tid > 0)
            fr_cpy(frtmp[dst], polynomial[src]);
        else
            fr_zero(frtmp[dst]);

        __syncthreads();

        // Zero the other half of coefficients before FFT

        fr_zero(frtmp[tid+1]); // should be fr_zero(frtmp[512*i+tid+1]); but illegal address!?

        // Compute FFT

        __syncthreads();
        fr_fft(frtmp, frtmp);
        __syncthreads();

        // multiply and accumulate

        g1p_cpy(t, xext_fft[512*i+tid+0]);
        g1p_mul(t, frtmp[tid]);
        __syncthreads();
        g1p_add(a0, t);

        g1p_cpy(t, xext_fft[512*i+tid+256]);
        g1p_mul(t, frtmp[tid+256]);
        __syncthreads();
        g1p_add(a1, t);
    }

    // Store accumulators

    g1p_cpy(g1ptmp[tid+  0], a0);
    g1p_cpy(g1ptmp[tid+256], a1);

    /// Part 3

    // Inverse G1 FFT
    g1p_ift(g1ptmp, g1ptmp);

    
    // Zero upper half of intermediate result
    g1p_inf(g1ptmp[256+tid]);

    // G1 FFT
    g1p_fft(h_fft, g1ptmp);

}

// vim: ts=4 et sw=4 si


    //debug, copy g1ptmp to output
//    g1p_cpy(h_fft[tid+  0], g1ptmp[tid+  0]);
//    g1p_cpy(h_fft[tid+256], g1ptmp[tid+256]);