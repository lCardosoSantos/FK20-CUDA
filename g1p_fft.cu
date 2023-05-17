#include <stdio.h>

#include "g1.cuh"
#include "fk20.cuh"

// Workspace in shared memory

extern __shared__ g1p_t g1p_tmp[];

// FFT over G1 with projective coordinates
// input and output may freely overlap

__device__ void g1p_fft(g1p_t *output, const g1p_t *input) {
    // One FFT of size 512 per thread block
    // No interleaving of data for different FFTs

    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned l, r, w, src, dst;

    // Adjust IO pointers to point at each thread block's data

    input  += 512*bid;
    output += 512*bid;

    // Copy inputs to workspace

    src = tid;
    // dst = 9 last bits of src reversed
    asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(dst) : "r"(src << (32-9)));

    g1p_cpy(g1p_tmp[dst], input[src]);

    src |= 256;
    dst |= 1;

    g1p_cpy(g1p_tmp[dst], input[src]);

    __syncthreads();

    //// Stage 0

    w = 0;
    l = 2 * tid;
    r = l | 1;

    //g1p_mul(g1p_tmp[r], fr_roots[w]);
    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);

    __syncthreads();

    //// Stage 1

    w = (tid & 1) << 7;
    l = tid + (tid & -2U);
    r = l | 2;

    if (w) g1p_mul(g1p_tmp[r], fr_roots[w]);
    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);

    __syncthreads();

    //// Stage 2

    w = (tid & 3) << 6;
    l = tid + (tid & -4U);
    r = l | 4;

    g1p_mul(g1p_tmp[r], fr_roots[w]);
    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);

    __syncthreads();

    //// Stage 3

    w = (tid & 7) << 5;
    l = tid + (tid & -8U);
    r = l | 8;

    g1p_mul(g1p_tmp[r], fr_roots[w]);
    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);

    __syncthreads();

    //// Stage 4

    w = (tid & 15) << 4;
    l = tid + (tid & -16U);
    r = l | 16;

    g1p_mul(g1p_tmp[r], fr_roots[w]);
    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);

    __syncthreads();

    //// Stage 5

    w = (tid & 31) << 3;
    l = tid + (tid & -32U);
    r = l | 32;

    g1p_mul(g1p_tmp[r], fr_roots[w]);
    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);

    __syncthreads();

    //// Stage 6

    w = (tid & 63) << 2;
    l = tid + (tid & -64U);
    r = l | 64;

    g1p_mul(g1p_tmp[r], fr_roots[w]);
    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);

    __syncthreads();

    //// Stage 7

    w = (tid & 127) << 1;
    l = tid + (tid & -128U);
    r = l | 128;

    g1p_mul(g1p_tmp[r], fr_roots[w]);
    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);

    __syncthreads();

    //// Stage 8

    w = (tid & 255) << 0;
    l = tid + (tid & -256U);
    r = l | 256;

    g1p_mul(g1p_tmp[r], fr_roots[w]);
    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);

    __syncthreads();

    // Copy results to output, no shuffle

    src = tid;
    dst = src;

    g1p_cpy(output[dst], g1p_tmp[src]);

    src += 256;
    dst += 256;

    g1p_cpy(output[dst], g1p_tmp[src]);
}

// Inverse FFT over G1 with projective coordinates

__device__ void g1p_ift(g1p_t *output, const g1p_t *input) {
    // One inverse FFT of size 512 per thread block
    // No interleaving of data for different FFTs

    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned l, r, w, src, dst;

    // Adjust IO pointers to point at each thread block's data

    input  += 512*bid;
    output += 512*bid;

    // Copy inputs to workspace, no shuffle

    src = tid;
    dst = src;

    g1p_cpy(g1p_tmp[dst], input[src]);

    src += 256;
    dst += 256;

    g1p_cpy(g1p_tmp[dst], input[src]);

    __syncthreads();

    //// Stage 8

    w = (tid & 255) << 0;
    l = tid + (tid & -256U);
    r = l | 256;

    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);
    g1p_mul(g1p_tmp[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 7

    w = (tid & 127) << 1;
    l = tid + (tid & -128U);
    r = l | 128;

    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);
    g1p_mul(g1p_tmp[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 6

    w = (tid & 63) << 2;
    l = tid + (tid & -64U);
    r = l | 64;

    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);
    g1p_mul(g1p_tmp[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 5

    w = (tid & 31) << 3;
    l = tid + (tid & -32U);
    r = l | 32;

    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);
    g1p_mul(g1p_tmp[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 4

    w = (tid & 15) << 4;
    l = tid + (tid & -16U);
    r = l | 16;

    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);
    g1p_mul(g1p_tmp[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 3

    w = (tid & 7) << 5;
    l = tid + (tid & -8U);
    r = l | 8;

    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);
    g1p_mul(g1p_tmp[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 2

    w = (tid & 3) << 6;
    l = tid + (tid & -4U);
    r = l | 4;

    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);
    g1p_mul(g1p_tmp[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 1

    w = (tid & 1) << 0;
    l = tid + (tid & -2U);
    r = l | 2;

    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);
    g1p_mul(g1p_tmp[l], fr_roots[513]);    // 2**-9
    g1p_mul(g1p_tmp[r], fr_roots[513+w]);  // w ? 2**-9/fr_roots[128] : 2**-9

    __syncthreads();

    //// Stage 0

    w = 0;
    l = 2 * tid;
    r = l | 1;

    g1p_addsub(g1p_tmp[l], g1p_tmp[r]);
    //g1p_mul(g1p_tmp[r], fr_roots[512-w]);

    __syncthreads();

    // Copy results to output

    dst = tid;
    // src = 9 last bits of dst reversed
    asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(src) : "r"(dst << (32-9)));

    g1p_cpy(output[dst], g1p_tmp[src]);

    dst |= 256;
    src |= 1;

    g1p_cpy(output[dst], g1p_tmp[src]);
}

// Kernel wrappers for device-side FFT functions

__global__ void g1p_fft_wrapper(g1p_t *output, const g1p_t *input) { g1p_fft(output, input); }
__global__ void g1p_ift_wrapper(g1p_t *output, const g1p_t *input) { g1p_ift(output, input); }

// vim: ts=4 et sw=4 si
