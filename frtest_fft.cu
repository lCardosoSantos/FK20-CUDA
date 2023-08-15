// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "frtest.cuh"

#include "fr_fft_testvector.cu"

__managed__ fr_t fft[512];
__managed__ uint8_t cmp[512];

/**
 * @brief Tests fft and inverse fft over Fr using KAT
 * 
 */
void FrTestFFT() {

    const size_t sharedmem = 512*4*8;   // 512 residues, 4 words/residue, 8 bytes/word
    cudaError_t err;

    bool pass = true;

    err = cudaFuncSetAttribute(fr_fft_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %d (%s)\n", err, cudaGetErrorName(err));

    err = cudaFuncSetAttribute(fr_ift_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %d (%s)\n", err, cudaGetErrorName(err));

    //////////////////////////////////////////////////

    printf("=== RUN   %s\n", "fr_fft");
    fr_fft_wrapper<<<1, 256, sharedmem>>>(fft, q);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_fft_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    fr_eq_wrapper<<<16, 32>>>(cmp, 512, fft, a);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check FFT result

    for (int i=0; pass && i<512; i++)
        if (cmp[i] != 1) {
            printf("FFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);

    //////////////////////////////////////////////////

    printf("=== RUN   %s\n", "fr_ift");
    fr_ift_wrapper<<<1, 256, sharedmem>>>(fft, a);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_ift_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    fr_eq_wrapper<<<16, 32>>>(cmp, 512, fft, q);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check IFT result

    for (int i=0; pass && i<512; i++)
        if (cmp[i] != 1) {
            printf("IFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);

    //////////////////////////////////////////////////
}

// vim: ts=4 et sw=4 si
