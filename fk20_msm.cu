// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <cassert>
#include <cstdio>

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

/**
 * @brief toeplitz_coefficients_fft + xext_fft -> hext_fft
 *
 * Grid must be 1-D, 256 threads per block.
 * WARN: Calling this function with dynamic shared memory introduces unpredictable behavior.
 *
 * @param[out] he_fft array with dimensions [gridDim.x * 512]
 * @param[in] tc_fft array with dimensions [gridDim.x * 16][512]
 * @param[in] xe_fft array with dimensions [16][512]
 * @return void
 */
__global__ void fk20_msm(g1p_t *he_fft, const fr_t *tc_fft, const g1p_t *xe_fft) {
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.x == 256);  // k
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block/row number

    g1p_t a0, a1, t;

    g1p_inf(a0);
    g1p_inf(a1);

    // Move pointer for blocks
    he_fft += 512*bid;
    tc_fft += 16*512*bid;

    __syncwarp();

    // MSM Loop
    for (int i=0; i<16; i++) {

        // Multiply and accumulate

        g1p_cpy(t, xe_fft[512*i+tid+0]);
        g1p_mul(t, tc_fft[512*i+tid+0]);
        g1p_add(a0, t);

        __syncwarp();

        g1p_cpy(t, xe_fft[512*i+tid+256]);
        g1p_mul(t, tc_fft[512*i+tid+256]);
        g1p_add(a1, t);
    }

    __syncwarp();

    // hext_fft = a0||a1
    // Store accumulators
    g1p_cpy(he_fft[tid+  0], a0);
    g1p_cpy(he_fft[tid+256], a1);
}

/**
 * @brief: fk20_msm_makelut(): xext_fft -> xext_lut
 *
 * Grid must be 8k single-threaded blocks.
 * This function precomputes the lookup tables for the comb multiplication.
 * It is not part of the time-critical pipeline of FK20 computations.
 * 
 * @param [out] xe_lut     G1a array with dimensions [16][512][256]
 * @param [in]  xe_fft     G1p array with dimensions [16][512]
 */

__global__ void fk20_msm_makelut(g1a_t xe_lut[16][512][256], const g1p_t xe_fft[16][512]) {
    assert(gridDim.x  == 512);
    assert(gridDim.y  ==  16);
    assert(gridDim.z  ==   1);
    assert(blockDim.x ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);

    __shared__ g1p_t lut[256];  // 36 KiB

    // Initialise all entries to inf

    for (int i=0; i<256; i++)
        g1p_inf(lut[i]);

    g1p_cpy(lut[0x01], xe_fft[blockIdx.y][blockIdx.x]);

    for (int i=2; i<256; i*=2) {
        g1p_cpy(lut[i], lut[i/2]);
        for (int d=0; d<32; d++)
            g1p_dbl(lut[i]);

        for (int j=1; j<i; j*=2) {
            g1p_cpy(lut[i|j], lut[i]);
            g1p_add(lut[i|j], lut[j]);

            for (int k=1; k<j; k*=2) {
                g1p_cpy(lut[i|j|k], lut[i|j]);
                g1p_add(lut[i|j|k], lut[k]);

                for (int l=1; l<k; l*=2) {
                    g1p_cpy(lut[i|j|k|l], lut[i|j|k]);
                    g1p_add(lut[i|j|k|l], lut[l]);

                    for (int m=1; m<l; m*=2) {
                        g1p_cpy(lut[i|j|k|l|m], lut[i|j|k|l]);
                        g1p_add(lut[i|j|k|l|m], lut[m]);

                        for (int n=1; n<m; n*=2) {
                            g1p_cpy(lut[i|j|k|l|m|n], lut[i|j|k|l|m]);
                            g1p_add(lut[i|j|k|l|m|n], lut[n]);

                            for (int o=1; o<n; o*=2) {
                                g1p_cpy(lut[i|j|k|l|m|n|o], lut[i|j|k|l|m|n]);
                                g1p_add(lut[i|j|k|l|m|n|o], lut[o]);

                                for (int p=1; p<o; p*=2) {
                                    g1p_cpy(lut[i|j|k|l|m|n|o|p], lut[i|j|k|l|m|n|o]);
                                    g1p_add(lut[i|j|k|l|m|n|o|p], lut[p]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Convert each table element to affine coordinates and store to xe_lut[]

    for (int i=0; i<256; i++)
        g1a_fromG1p(xe_lut[blockIdx.y][blockIdx.x][i], lut[i]);
}

// Flip the bit order, then transpose each 8x8 bit matrix
__device__ void multiplier_reorg(uint32_t z[8], const uint32_t x[8]) {
    asm volatile (
        "\n\t{"
        "\n\t.reg .u32 z<8>, x<8>, t0;"

        // Load input words, reverse their bits

        "\n\tld.cs.b32 x0, [%1+0x00];\n\tbrev.b32 x0, x0;"
        "\n\tld.cs.b32 x1, [%1+0x04];\n\tbrev.b32 x1, x1;"
        "\n\tld.cs.b32 x2, [%1+0x08];\n\tbrev.b32 x2, x2;"
        "\n\tld.cs.b32 x3, [%1+0x0c];\n\tbrev.b32 x3, x3;"
        "\n\tld.cs.b32 x4, [%1+0x10];\n\tbrev.b32 x4, x4;"
        "\n\tld.cs.b32 x5, [%1+0x14];\n\tbrev.b32 x5, x5;"
        "\n\tld.cs.b32 x6, [%1+0x18];\n\tbrev.b32 x6, x6;"
        "\n\tld.cs.b32 x7, [%1+0x1c];\n\tbrev.b32 x7, x7;"

        // Transpose

        "\n\tand.b32 z0, x0, 0x01010101;"
        "\n\tand.b32 t0, x1, 0x01010101;\n\tshl.b32 t0, t0, 1;\n\tor.b32 z0, z0, t0;"
        "\n\tand.b32 t0, x2, 0x01010101;\n\tshl.b32 t0, t0, 2;\n\tor.b32 z0, z0, t0;"
        "\n\tand.b32 t0, x3, 0x01010101;\n\tshl.b32 t0, t0, 3;\n\tor.b32 z0, z0, t0;"
        "\n\tand.b32 t0, x4, 0x01010101;\n\tshl.b32 t0, t0, 4;\n\tor.b32 z0, z0, t0;"
        "\n\tand.b32 t0, x5, 0x01010101;\n\tshl.b32 t0, t0, 5;\n\tor.b32 z0, z0, t0;"
        "\n\tand.b32 t0, x6, 0x01010101;\n\tshl.b32 t0, t0, 6;\n\tor.b32 z0, z0, t0;"
        "\n\tand.b32 t0, x7, 0x01010101;\n\tshl.b32 t0, t0, 7;\n\tor.b32 z0, z0, t0;"

        "\n\tand.b32 z1, x1, 0x02020202;"
        "\n\tand.b32 t0, x2, 0x02020202;\n\tshl.b32 t0, t0, 1;\n\tor.b32 z1, z1, t0;"
        "\n\tand.b32 t0, x3, 0x02020202;\n\tshl.b32 t0, t0, 2;\n\tor.b32 z1, z1, t0;"
        "\n\tand.b32 t0, x4, 0x02020202;\n\tshl.b32 t0, t0, 3;\n\tor.b32 z1, z1, t0;"
        "\n\tand.b32 t0, x5, 0x02020202;\n\tshl.b32 t0, t0, 4;\n\tor.b32 z1, z1, t0;"
        "\n\tand.b32 t0, x6, 0x02020202;\n\tshl.b32 t0, t0, 5;\n\tor.b32 z1, z1, t0;"
        "\n\tand.b32 t0, x7, 0x02020202;\n\tshl.b32 t0, t0, 6;\n\tor.b32 z1, z1, t0;"
        "\n\tand.b32 t0, x0, 0x02020202;\n\tshr.b32 t0, t0, 1;\n\tor.b32 z1, z1, t0;"

        "\n\tand.b32 z2, x2, 0x04040404;"
        "\n\tand.b32 t0, x3, 0x04040404;\n\tshl.b32 t0, t0, 1;\n\tor.b32 z2, z2, t0;"
        "\n\tand.b32 t0, x4, 0x04040404;\n\tshl.b32 t0, t0, 2;\n\tor.b32 z2, z2, t0;"
        "\n\tand.b32 t0, x5, 0x04040404;\n\tshl.b32 t0, t0, 3;\n\tor.b32 z2, z2, t0;"
        "\n\tand.b32 t0, x6, 0x04040404;\n\tshl.b32 t0, t0, 4;\n\tor.b32 z2, z2, t0;"
        "\n\tand.b32 t0, x7, 0x04040404;\n\tshl.b32 t0, t0, 5;\n\tor.b32 z2, z2, t0;"
        "\n\tand.b32 t0, x0, 0x04040404;\n\tshr.b32 t0, t0, 2;\n\tor.b32 z2, z2, t0;"
        "\n\tand.b32 t0, x1, 0x04040404;\n\tshr.b32 t0, t0, 1;\n\tor.b32 z2, z2, t0;"

        "\n\tand.b32 z3, x3, 0x08080808;"
        "\n\tand.b32 t0, x4, 0x08080808;\n\tshl.b32 t0, t0, 1;\n\tor.b32 z3, z3, t0;"
        "\n\tand.b32 t0, x5, 0x08080808;\n\tshl.b32 t0, t0, 2;\n\tor.b32 z3, z3, t0;"
        "\n\tand.b32 t0, x6, 0x08080808;\n\tshl.b32 t0, t0, 3;\n\tor.b32 z3, z3, t0;"
        "\n\tand.b32 t0, x7, 0x08080808;\n\tshl.b32 t0, t0, 4;\n\tor.b32 z3, z3, t0;"
        "\n\tand.b32 t0, x0, 0x08080808;\n\tshr.b32 t0, t0, 3;\n\tor.b32 z3, z3, t0;"
        "\n\tand.b32 t0, x1, 0x08080808;\n\tshr.b32 t0, t0, 2;\n\tor.b32 z3, z3, t0;"
        "\n\tand.b32 t0, x2, 0x08080808;\n\tshr.b32 t0, t0, 1;\n\tor.b32 z3, z3, t0;"

        "\n\tand.b32 z4, x4, 0x10101010;"
        "\n\tand.b32 t0, x5, 0x10101010;\n\tshl.b32 t0, t0, 1;\n\tor.b32 z4, z4, t0;"
        "\n\tand.b32 t0, x6, 0x10101010;\n\tshl.b32 t0, t0, 2;\n\tor.b32 z4, z4, t0;"
        "\n\tand.b32 t0, x7, 0x10101010;\n\tshl.b32 t0, t0, 3;\n\tor.b32 z4, z4, t0;"
        "\n\tand.b32 t0, x0, 0x10101010;\n\tshr.b32 t0, t0, 4;\n\tor.b32 z4, z4, t0;"
        "\n\tand.b32 t0, x1, 0x10101010;\n\tshr.b32 t0, t0, 3;\n\tor.b32 z4, z4, t0;"
        "\n\tand.b32 t0, x2, 0x10101010;\n\tshr.b32 t0, t0, 2;\n\tor.b32 z4, z4, t0;"
        "\n\tand.b32 t0, x3, 0x10101010;\n\tshr.b32 t0, t0, 1;\n\tor.b32 z4, z4, t0;"

        "\n\tand.b32 z5, x5, 0x20202020;"
        "\n\tand.b32 t0, x6, 0x20202020;\n\tshl.b32 t0, t0, 1;\n\tor.b32 z5, z5, t0;"
        "\n\tand.b32 t0, x7, 0x20202020;\n\tshl.b32 t0, t0, 2;\n\tor.b32 z5, z5, t0;"
        "\n\tand.b32 t0, x0, 0x20202020;\n\tshr.b32 t0, t0, 5;\n\tor.b32 z5, z5, t0;"
        "\n\tand.b32 t0, x1, 0x20202020;\n\tshr.b32 t0, t0, 4;\n\tor.b32 z5, z5, t0;"
        "\n\tand.b32 t0, x2, 0x20202020;\n\tshr.b32 t0, t0, 3;\n\tor.b32 z5, z5, t0;"
        "\n\tand.b32 t0, x3, 0x20202020;\n\tshr.b32 t0, t0, 2;\n\tor.b32 z5, z5, t0;"
        "\n\tand.b32 t0, x4, 0x20202020;\n\tshr.b32 t0, t0, 1;\n\tor.b32 z5, z5, t0;"

        "\n\tand.b32 z6, x6, 0x40404040;"
        "\n\tand.b32 t0, x7, 0x40404040;\n\tshl.b32 t0, t0, 1;\n\tor.b32 z6, z6, t0;"
        "\n\tand.b32 t0, x0, 0x40404040;\n\tshr.b32 t0, t0, 6;\n\tor.b32 z6, z6, t0;"
        "\n\tand.b32 t0, x1, 0x40404040;\n\tshr.b32 t0, t0, 5;\n\tor.b32 z6, z6, t0;"
        "\n\tand.b32 t0, x2, 0x40404040;\n\tshr.b32 t0, t0, 4;\n\tor.b32 z6, z6, t0;"
        "\n\tand.b32 t0, x3, 0x40404040;\n\tshr.b32 t0, t0, 3;\n\tor.b32 z6, z6, t0;"
        "\n\tand.b32 t0, x4, 0x40404040;\n\tshr.b32 t0, t0, 2;\n\tor.b32 z6, z6, t0;"
        "\n\tand.b32 t0, x5, 0x40404040;\n\tshr.b32 t0, t0, 1;\n\tor.b32 z6, z6, t0;"

        "\n\tand.b32 z7, x7, 0x80808080;"
        "\n\tand.b32 t0, x0, 0x80808080;\n\tshr.b32 t0, t0, 7;\n\tor.b32 z7, z7, t0;"
        "\n\tand.b32 t0, x1, 0x80808080;\n\tshr.b32 t0, t0, 6;\n\tor.b32 z7, z7, t0;"
        "\n\tand.b32 t0, x2, 0x80808080;\n\tshr.b32 t0, t0, 5;\n\tor.b32 z7, z7, t0;"
        "\n\tand.b32 t0, x3, 0x80808080;\n\tshr.b32 t0, t0, 4;\n\tor.b32 z7, z7, t0;"
        "\n\tand.b32 t0, x4, 0x80808080;\n\tshr.b32 t0, t0, 3;\n\tor.b32 z7, z7, t0;"
        "\n\tand.b32 t0, x5, 0x80808080;\n\tshr.b32 t0, t0, 2;\n\tor.b32 z7, z7, t0;"
        "\n\tand.b32 t0, x6, 0x80808080;\n\tshr.b32 t0, t0, 1;\n\tor.b32 z7, z7, t0;"

        "\n\tst.b32 [%0+0x00], z0;"
        "\n\tst.b32 [%0+0x04], z1;"
        "\n\tst.b32 [%0+0x08], z2;"
        "\n\tst.b32 [%0+0x0c], z3;"
        "\n\tst.b32 [%0+0x10], z4;"
        "\n\tst.b32 [%0+0x14], z5;"
        "\n\tst.b32 [%0+0x18], z6;"
        "\n\tst.b32 [%0+0x1c], z7;"

        "\n\t"
        "}\n\t"
        :: "l"(z), "l"(x) : "memory"
    );
}

/**
 * @brief toeplitz_coefficients_fft + xext_fft -> hext_fft
 * Performs the same operation as fk20_msm(), but column by column
 * instead of row by row. This, combined with xe_fft being constant
 * for a given setup, permits the use of 8-way comb multiplication.
 * Since all point multiplications in one column use the same element
 * of xe_fft, fk20_msm_makelut() precomputes sums of $P*2^i$ for $i$
 * a multiple of 32 from 0 to 224.
 *
 * @param[out] he_fft G1p array with dimensions [512]    [512]
 * @param[in]  tc_fft Fr  array with dimensions [512][16][512]
 * @param[in]  xe_lut G1a array with dimensions      [16][512][256]
 * @return void
 */
__global__ void fk20_msm_comb(g1p_t he_fft[512][512], const fr_t tc_fft[512][16][512], const g1a_t xe_lut[16][512][256]) {
    assert(gridDim.x  == 512);  // Number of MSMs (16-element columns) to process per row
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.x == 256);  // Rows/2.
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);

    unsigned tid = threadIdx.x; // Thread/row number
    unsigned bid = blockIdx.x;  // Block/column number

    // TODO: Change lut[] to g1a_t and pad each element to 25 words (100 instead of 96 bytes) to reduce bank conflicts.

    __shared__ g1p_t lut[256];          // Lookup table for all threads. 36 KiB, statically allocated.
    __shared__ uint32_t mul[256][9];    // Multipliers, one per thread, padded to avoid bank conflicts. 9 KiB, statically allocated.

    g1p_t sum0, sum1, t0, t1; // Running sums and temporaries in local (thread-interleaved global) memory

    // Initialise running sums

    g1p_inf(sum0);
    g1p_inf(sum1);

    for (unsigned i=0; i<16; i++) {

        // Copy lookup table
        g1p_fromG1a(lut[tid], xe_lut[i][bid][tid]);

        g1p_inf(t0);
        g1p_inf(t1);

        // Load first multiplier from global memory, reorganise bits and store in shared memory
        multiplier_reorg(mul[tid], (uint32_t *)&(tc_fft[tid][i][bid]));

        __syncthreads();

        for (int j=0; j<32; j++) {
            int word = j & 7;

            g1p_dbl(t0);
            g1p_add(t0, lut[ 0xff & mul[tid][word] ]);

            // g1p_msm_multi(-4, &t0, NULL, &t0, &lut[ 0xff & mul[tid][word] ]);

            mul[tid][word] >>= 8;
        }

        __syncthreads();

        // Load second multiplier from global memory, reorganise bits and store in shared memory
        multiplier_reorg(mul[tid], (uint32_t *)&(tc_fft[tid+256][i][bid]));

        __syncthreads();

        for (int j=0; j<32; j++) {
            int word = j & 7;

            g1p_dbl(t1);
            g1p_add(t1, lut[ 0xff & mul[tid][word] ]);

            // g1p_msm_multi(-4, &t1, NULL, &t1, &lut[ 0xff & mul[tid][word] ]);

            mul[tid][word] >>= 8;
        }
        __syncthreads();

        g1p_add(sum0, t0);
        g1p_add(sum1, t1);
    }

    // Store results
    g1p_cpy(he_fft[tid+  0][bid], sum0);
    g1p_cpy(he_fft[tid+256][bid], sum1);
}


/*                 cudaGraph implementation                                   */

//copy of base case for debugging.
__global__ void fk20_msm_comb_tmp(g1p_t he_fft[512][512], const fr_t tc_fft[512][16][512], const g1a_t xe_lut[16][512][256], unsigned col);

#define cudaErrCheck(fmt, ...)                                                                                         \
    if (err != cudaSuccess)                                                                                            \
    printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)

#define DEBUG

#ifdef DEBUG
    #define dprintf(...) fprintf(stderr, ##__VA_ARGS__)
#else
    #define dprintf(...)
#endif

//Control variables
bool msmGraphCreated = false;
g1p_t* msmGraphArgs[3];
cudaGraph_t msmGraph;
cudaGraphExec_t msmGraphExec;

/**
 * @brief 
 *
 * @param[out] he_fft G1p array with dimensions [512]    [512]
 * @param[in]  tc_fft Fr  array with dimensions [512][16][512]
 * @param[in]  xe_lut G1a array with dimensions      [16][512][256]
 * @return void
 */
void fk20_msm_comb_graph(g1p_t he_fft[512][512], const fr_t tc_fft[512][16][512], const g1a_t xe_lut[16][512][256]){
    cudaError_t err; 
    cudaStream_t zeroStream;
    cudaStreamCreate(&zeroStream);

    const unsigned nCols = 512;
    const unsigned nThreads = 256; 
    const unsigned nBlocks = 2; 

    

    // TODO: Check for parameters too
    if (!msmGraphCreated) {
        dprintf("Graph init\n");
        cudaStream_t sZero;
        cudaStreamCreate(&sZero);

        cudaEvent_t forkEvent, joinEvent[nCols];
        cudaEventCreate(&forkEvent);
        for (unsigned i = 0; i < nCols; i++) {
            cudaEventCreate(&joinEvent[i]);
        }

        cudaStream_t colStreams[nCols];
        for (unsigned i = 0; i < nCols; i++) {
            cudaStreamCreate(&colStreams[i]);
        }

        // Start graph capture
        cudaStreamBeginCapture(sZero, cudaStreamCaptureModeGlobal);
        // Fork graph
        cudaEventRecord(forkEvent, sZero);

        ////////////////////////////////////////////////////////////////////////
        for (int i = 0; i < 512; i++) {
            cudaStreamWaitEvent(colStreams[i], forkEvent);
            fk20_msm_comb_tmp<<<nBlocks, nThreads, 0, colStreams[i]>>>(he_fft, tc_fft, xe_lut, i);
            cudaEventRecord(joinEvent[i],
                            colStreams[i]); // Join all streams to sZero
        }
        ////////////////////////////////////////////////////////////////////////

        // Join graph
        for (unsigned i = 0; i < nCols; i++)
            cudaStreamWaitEvent(sZero, joinEvent[i]);

        // End graph capture
        cudaStreamEndCapture(sZero, &msmGraph);
        err = cudaGraphInstantiate(&msmGraphExec, msmGraph, 0);
        cudaErrCheck("graph instantiate");

        // destroy stream
        cudaStreamDestroy(sZero);
        for (unsigned i = 0; i < nCols; i++) {
            cudaStreamDestroy(colStreams[i]);
        }

        // destroy events
        cudaEventDestroy(forkEvent);
        for (unsigned i = 0; i < nCols; i++)
            cudaEventDestroy(joinEvent[i]);
        
        msmGraphCreated = true;
    }

    dprintf("Graph launch\n");
    err = cudaGraphLaunch(msmGraphExec, zeroStream); 
    cudaErrCheck("graph launch");
}

__global__ void fk20_msm_comb_tmp(g1p_t he_fft[512][512], const fr_t tc_fft[512][16][512], const g1a_t xe_lut[16][512][256], unsigned col) {
    // if (gridDim.x  != 512) return;  // Number of MSMs (16-element columns) to process per row
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    // if (blockDim.x != 256) return;  // Rows/2.
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread/row number
    unsigned bid = blockIdx.x;  // Block/column number
    unsigned idx = blockDim.x*bid+tid;

    __shared__ g1p_t lut[256];          // Lookup table for all threads. 36 KiB, statically allocated.
    __shared__ uint32_t mul[256][9];    // Multipliers, one per thread, padded to avoid bank conflicts. 9 KiB, statically allocated.

    g1p_t sum0, t0; // Running sums and temporaries in local (thread-interleaved global) memory

    // Initialise running sums
    g1p_inf(sum0);

    for (unsigned i=0; i<16; i++) {

        // Copy lookup table
        g1p_fromG1a(lut[tid], xe_lut[i][col][tid]);

        g1p_inf(t0);

        // Load first multiplier from global memory, reorganise bits and store in shared memory
        multiplier_reorg(mul[tid], (uint32_t *)&(tc_fft[idx][i][col]));

        __syncthreads();

        for (int j=0; j<32; j++) {
            int word = j & 7;

            g1p_dbl(t0);
            g1p_add(t0, lut[ 0xff & mul[tid][word] ]);

            // g1p_msm_multi(-4, &t0, NULL, &t0, &lut[ 0xff & mul[idx][word] ]);

            mul[tid][word] >>= 8;
        }

        __syncthreads();

        g1p_add(sum0, t0);
    }

    // Store results
    g1p_cpy(he_fft[idx+  0][col], sum0);
}

// vim: ts=4 et sw=4 si
