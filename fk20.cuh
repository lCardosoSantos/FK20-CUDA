// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FK20_CUH
#define FK20_CUH

#include <stdint.h>

#include "fr.cuh"
#include "g1.cuh"

// Shared memory sizes
const size_t g1p_sharedmem = 512 * 3 * 6 * 8; // 512 points * 3 residues/point * 6 words/residue * 8 bytes/word = 72 KiB
const size_t fr_sharedmem = 512 * 4 * 8;      // 512 residues * 4 words/residue * 8 bytes/word = 16 KiB

// External interface
// Consult README file for definitions of nomenclature.

__global__ void fk20_setup2xext_fft(g1p_t xext_fft[8192], const g1p_t *setup);

__host__ void fk20_poly2h_fft(g1p_t *h_fft, const fr_t *polynomial, const g1p_t xext_fft[8192], unsigned rows);

__global__ void fk20_poly2toeplitz_coefficients(fr_t *toeplitz_coefficients, const fr_t *polynomial);
__global__ void fk20_poly2toeplitz_coefficients_fft(fr_t *toeplitz_coefficients_fft, const fr_t *polynomial);
__global__ void fk20_poly2hext_fft(g1p_t *hext_fft, const fr_t *polynomial, const g1p_t xext_fft[8192]);

__global__ void fk20_hext_fft2h(g1p_t *h, const g1p_t *hext_fft);
__global__ void fk20_h2h_fft(g1p_t *h_fft, const g1p_t *h);
__global__ void fk20_hext_fft2h_fft(g1p_t *h_fft, const g1p_t *hext_fft);
__global__ void fk20_hext2h(g1p_t *h);
__global__ void fk20_msm(g1p_t *hext_fft, const fr_t *toeplitz_coefficients_fft, const g1p_t *xext_fft);

// useful macros. Need to have a cudaError_t err variable declared in the caller

// Syncronizes the Device, making sure that the kernel has finished the execution. Checks for any errors, and report if
// errors are found.
#ifndef CUDASYNC
#define CUDASYNC(fmt, ...)                                                                                             \
    err = cudaDeviceSynchronize();                                                                                     \
    if (err != cudaSuccess)                                                                                            \
    printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)
#endif

// The necessary shared memory is larger than what we can statically allocate, hence it is allocated dynamically in the
// kernel call. Because cuda, we need to set the maximum allowed size using this macro.
#ifndef SET_SHAREDMEM
#define SET_SHAREDMEM(SZ, FN)                                                                                          \
    err = cudaFuncSetAttribute(FN, cudaFuncAttributeMaxDynamicSharedMemorySize, SZ);                                   \
    cudaDeviceSynchronize();                                                                                           \
    if (err != cudaSuccess)                                                                                            \
        printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
#endif


#endif

// vim: ts=4 et sw=4 si
