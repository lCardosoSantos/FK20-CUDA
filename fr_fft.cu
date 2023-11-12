// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "fk20.cuh"

/**
 * @brief  Workspace in shared memory. Must be 512*sizeof(fr_t) bytes
 * 
 */
extern __shared__ fr_t fr_smem[];

/**
 * @brief FFT over Fr
 * 
 * Performs one FFT-512 for each thread block.
 * This function must be called with 256 threads per block, i.e. dim3(256,1,1).
 * Input and output arrays can overlap without side effects.
 * There is no interleaving of data for different FFTs (the stride is 1).
 * 
 * @param[out] output 
 * @param[in] input 
 * @return void 
 */
__device__ void fr_fft(fr_t *output, const fr_t *input) {

    unsigned tid = threadIdx.x; // Thread number
    unsigned l, r, w, src, dst;

    // Copy inputs to workspace

    src = tid;
    // dst = 9 last bits of src reversed
    asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(dst) : "r"(src << (32-9)));

    fr_cpy(fr_smem[dst], input[src]);

    src |= 256;
    dst |= 1;

    fr_cpy(fr_smem[dst], input[src]);

    __syncthreads();

    //// Stage 0

    w = 0;
    l = 2 * tid;
    r = l | 1;

    //fr_mul(fr_smem[r], fr_roots[w]);
    fr_addsub(fr_smem[l], fr_smem[r]);

    __syncwarp();

    //// Stage 1

    w = (tid & 1) << 7;
    l = tid + (tid & -2U);
    r = l | 2;

    if (w) fr_mul(fr_smem[r], fr_roots[w]);
    fr_addsub(fr_smem[l], fr_smem[r]);

    __syncwarp();

    //// Stage 2

    w = (tid & 3) << 6;
    l = tid + (tid & -4U);
    r = l | 4;

    fr_mul(fr_smem[r], fr_roots[w]);
    fr_addsub(fr_smem[l], fr_smem[r]);

    __syncwarp();

    //// Stage 3

    w = (tid & 7) << 5;
    l = tid + (tid & -8U);
    r = l | 8;

    fr_mul(fr_smem[r], fr_roots[w]);
    fr_addsub(fr_smem[l], fr_smem[r]);

    __syncwarp();

    //// Stage 4

    w = (tid & 15) << 4;
    l = tid + (tid & -16U);
    r = l | 16;

    fr_mul(fr_smem[r], fr_roots[w]);
    fr_addsub(fr_smem[l], fr_smem[r]);

    __syncwarp();

    //// Stage 5

    w = (tid & 31) << 3;
    l = tid + (tid & -32U);
    r = l | 32;

    fr_mul(fr_smem[r], fr_roots[w]);
    fr_addsub(fr_smem[l], fr_smem[r]);

    __syncthreads();

    //// Stage 6

    w = (tid & 63) << 2;
    l = tid + (tid & -64U);
    r = l | 64;

    fr_mul(fr_smem[r], fr_roots[w]);
    fr_addsub(fr_smem[l], fr_smem[r]);

    __syncthreads();

    //// Stage 7

    w = (tid & 127) << 1;
    l = tid + (tid & -128U);
    r = l | 128;

    fr_mul(fr_smem[r], fr_roots[w]);
    fr_addsub(fr_smem[l], fr_smem[r]);

    __syncthreads();

    //// Stage 8

    w = (tid & 255) << 0;
    l = tid + (tid & -256U);
    r = l | 256;

    fr_mul(fr_smem[r], fr_roots[w]);
    fr_addsub(fr_smem[l], fr_smem[r]);

    __syncthreads();

    // Copy results to output, no shuffle

    src = tid;
    dst = src;

    fr_cpy(output[dst], fr_smem[src]);

    src += 256;
    dst += 256;

    fr_cpy(output[dst], fr_smem[src]);
}

/**
 * @brief Inverse FFT for fr_t[512]
 * 
 * Performs one inverse FFT-512 in each thread block.
 * This function must be called with 256 threads per block, i.e. dim3(256,1,1).
 * Input and output arrays can overlap without side effects.
 * There is no interleaving of data for different FFTs (the stride is 1).
 * 
 * @param[out] output 
 * @param[in] input 
 * @return void 
 */
__device__ void fr_ift(fr_t *output, const fr_t *input) {

    unsigned tid = threadIdx.x; // Thread number
    unsigned l, r, w, src, dst;

    // Copy inputs to workspace, no shuffle

    src = tid;
    dst = src;

    fr_cpy(fr_smem[dst], input[src]);

    src += 256;
    dst += 256;

    fr_cpy(fr_smem[dst], input[src]);

    __syncthreads();

    //// Stage 8

    w = (tid & 255) << 0;
    l = tid + (tid & -256U);
    r = l | 256;

    fr_addsub(fr_smem[l], fr_smem[r]);
    fr_mul(fr_smem[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 7

    w = (tid & 127) << 1;
    l = tid + (tid & -128U);
    r = l | 128;

    fr_addsub(fr_smem[l], fr_smem[r]);
    fr_mul(fr_smem[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 6

    w = (tid & 63) << 2;
    l = tid + (tid & -64U);
    r = l | 64;

    fr_addsub(fr_smem[l], fr_smem[r]);
    fr_mul(fr_smem[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 5

    w = (tid & 31) << 3;
    l = tid + (tid & -32U);
    r = l | 32;

    fr_addsub(fr_smem[l], fr_smem[r]);
    fr_mul(fr_smem[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 4

    w = (tid & 15) << 4;
    l = tid + (tid & -16U);
    r = l | 16;

    fr_addsub(fr_smem[l], fr_smem[r]);
    fr_mul(fr_smem[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 3

    w = (tid & 7) << 5;
    l = tid + (tid & -8U);
    r = l | 8;

    fr_addsub(fr_smem[l], fr_smem[r]);
    fr_mul(fr_smem[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 2

    w = (tid & 3) << 6;
    l = tid + (tid & -4U);
    r = l | 4;

    fr_addsub(fr_smem[l], fr_smem[r]);
    fr_mul(fr_smem[r], fr_roots[512-w]);

    __syncthreads();

    //// Stage 1

    w = (tid & 1) << 0;
    l = tid + (tid & -2U);
    r = l | 2;

    fr_addsub(fr_smem[l], fr_smem[r]);
    fr_mul(fr_smem[l], fr_roots[513]);      // 2**-9
    fr_mul(fr_smem[r], fr_roots[513+w]);    // w ? 2**-9/fr_roots[128] : 2**-9

    __syncthreads();

    //// Stage 0

    w = 0;
    l = 2 * tid;
    r = l | 1;

    fr_addsub(fr_smem[l], fr_smem[r]);
    //fr_mul(fr_smem[r], fr_roots[512-w]);

    __syncthreads();

    // Copy results to output

    dst = tid;
    // src = 9 last bits of dst reversed
    asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(src) : "r"(dst << (32-9)));

    fr_cpy(output[dst], fr_smem[src]);

    dst |= 256;
    src |= 1;

    fr_cpy(output[dst], fr_smem[src]);
}

// Kernel wrappers for device-side FFT functions

/**
 * @brief wrapper for fr_fft: FFT for fr_t[512]
 * 
 * Executes an FFT over many arrays fr_t[512]. One array per block. input and 
 * output can overlap without side effects. There is no interleaving of data for
 * different FFTs.
 * 
 * @param[out] output 
 * @param[in] input 
 * @return void 
 */
__global__ void fr_fft_wrapper(fr_t *output, const fr_t *input) {

    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    // Adjust IO pointers to point at each thread block's data

    unsigned bid = blockIdx.x;  // Block number

    input  += 512*bid;
    output += 512*bid;

    fr_fft(output, input);
}

/**
 * @brief wrapper for fr_ift: inverse FFT for fr_t[512]
 * 
 * Executes an inverse FFT over many arrays fr_t[512]. One array per block. input and 
 * output can overlap without side effects. There is no interleaving of data for
 * different iFFTs.
 * 
 * @param[out] output 
 * @param[in] input 
 * @return void 
 */
__global__ void fr_ift_wrapper(fr_t *output, const fr_t *input) {

    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    // Adjust IO pointers to point at each thread block's data

    unsigned bid = blockIdx.x;  // Block number

    input  += 512*bid;
    output += 512*bid;

    fr_ift(output, input);
}

// vim: ts=4 et sw=4 si
