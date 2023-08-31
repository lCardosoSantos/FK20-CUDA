// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "g1.cuh"
#include "fk20.cuh"

// Workspace in shared mem
extern __shared__ g1p_t g1p_tmp[]; //512 * 3 * 6 * 8; // 512 points * 3 residues/point * 6 words/residue * 8 bytes/word = 72 KiB

__device__ void fft();
__device__ void ift();

__global__ void fk20_hext_fft2h_fft(g1p_t *h_fft, const g1p_t *hext_fft){
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    //Move pointer to block
    hext_fft  += 512*bid;
    h_fft += 512*bid;

    // STEP1 hext_fft -> hext 
    g1p_cpy(g1p_tmp[tid], hext_fft[tid]);
    g1p_cpy(g1p_tmp[tid+256], hext_fft[tid+256]);
    // IFT of sharedmem
    ift();

    // STEP2 hext -> h
    // Zeroing second half
    g1p_inf(g1p_tmp[256+tid]);

    // STEP3 h -> h_fft
    // FFT of sharedmem
    fft();

    //move from shared mem into h_fft
    g1p_cpy(h_fft[tid], g1p_tmp[tid]);
    g1p_cpy(h_fft[tid+256], g1p_tmp[tid+256]);
}

__device__ void ift(){
    unsigned src, dst;
    unsigned l, r, w, tid;

    tid = threadIdx.x;
    src = threadIdx.x;
    dst = src;

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
    //g1p_mul(output[r], fr_roots[512-w]);

    __syncthreads();

    // Last move
    dst = threadIdx.x;
    // src = 9 last bits of dst reversed
    asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(src) : "r"(dst << (32-9)));

    g1p_t m;

    g1p_cpy(m, g1p_tmp[dst]);
    g1p_cpy(g1p_tmp[dst], g1p_tmp[src]);
    g1p_cpy(g1p_tmp[src], m); 

    dst |= 256;
    src |= 1;

    g1p_cpy(m, g1p_tmp[dst]);
    g1p_cpy(g1p_tmp[dst], g1p_tmp[src]);
    g1p_cpy(g1p_tmp[src], m); 



}


__device__ void fft(){
    unsigned src, dst;
    unsigned l, r, w, tid;
    
    tid = threadIdx.x;
    src = threadIdx.x;
    // dst = 9 last bits of src reversed
    asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(dst) : "r"(src << (32-9)));

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
}

