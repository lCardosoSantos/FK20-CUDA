// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fr.cuh"
#include "g1.cuh"
#include "g1test.cuh"

__managed__ fr_t  X[512*512], Y[512*512], Z[512*512];
__managed__ g1p_t P[512*512], Q[512*512], R[512*512], S[512*512], T[512*512];

__managed__ uint8_t cmp[512*512];

////////////////////////////////////////////////////////////

#define SET_SHAREDMEM(SZ, FN) \
    err = cudaFuncSetAttribute(FN, cudaFuncAttributeMaxDynamicSharedMemorySize, SZ); \
    cudaDeviceSynchronize(); \
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

#define CUDASYNC     err = cudaDeviceSynchronize(); \
                     if (err != cudaSuccess) printf("Error: %d (%s)\n", err, cudaGetErrorName(err))

#define CUDASYNC     err = cudaDeviceSynchronize(); \
                     if (err != cudaSuccess) printf("Error: %d (%s)\n", err, cudaGetErrorName(err))

////////////////////////////////////////////////////////////

__global__ void g1p_add_wrapper(g1p_t *sum, int count, const g1p_t *x, const g1p_t *y) {

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    unsigned step = gridDim.z * gridDim.y * gridDim.x
                * blockDim.z * blockDim.y * blockDim.x;

    for (unsigned i=tid; i<count; i+=step) {
        g1p_t p;
        g1p_cpy(p, x[i]);
        g1p_add(p, y[i]);
        g1p_cpy(sum[i], p);
    }
}

////////////////////////////////////////////////////////////

__global__ void g1p_mul_wrapper(g1p_t *q, int count, const fr_t *x, const g1p_t *p) {

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    unsigned step = gridDim.z * gridDim.y * gridDim.x
                * blockDim.z * blockDim.y * blockDim.x;

    for (unsigned i=tid; i<count; i+=step) {
        g1p_t t;
        g1p_cpy(t, p[i]);
        g1p_mul(t, x[i]);
        g1p_cpy(q[i], t);
    }
}

////////////////////////////////////////////////////////////

__global__ void g1p_fr2g1p_wrapper(g1p_t *g1, int count, const fr_t *fr) {

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    unsigned step = gridDim.z * gridDim.y * gridDim.x
                * blockDim.z * blockDim.y * blockDim.x;

    for (unsigned i=tid; i<count; i+=step) {
        g1p_t p;
        g1p_gen(p);
        g1p_mul(p, fr[i]);
        g1p_cpy(g1[i], p);
    }
}

////////////////////////////////////////////////////////////

void G1TestFFT(unsigned rows) {
    const char filename[] = "/dev/urandom";
    FILE *pf = NULL;
    size_t result = 0;
    cudaError_t err = cudaSuccess;
    bool pass = true;
    int i;

    // Shared memory sizes

    const size_t g1p_sharedmem = 512*3*6*8; // 512 points * 3 residues/point * 6 words/residue * 8 bytes/word = 72 KiB
    const size_t fr_sharedmem = 512*4*8; // 512 residues * 4 words/residue * 8 bytes/word = 16 KiB

    // Setup

    SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    // Initialise X with random values

    pf = fopen(filename, "r");

    if (!pf) {
        fprintf(stderr, "Error opening %s\n", filename);
        return;
    }

    // printf("Initialising X[]\n");

    result = fread(X, sizeof(fr_t), rows*512, pf);

    if (result < rows*512) {
        fprintf(stderr, "Only read %zd values\n", result);
        goto L_pf;
    }

    // printf("Initialising P[]\n");

    // Initialise P: P[i] = X[i] * G

    g1p_fr2g1p_wrapper<<<32, 256>>>(P, rows*512, X);

    // printf("Initialising Y[]\n");

    result = fread(Y, sizeof(fr_t), rows*512, pf);

    if (result < rows*512) {
        fprintf(stderr, "Only read %zd values\n", result);
        goto L_pf;
    }

    // printf("Initialising Q[]\n");

    // Initialise Q: Q[i] = Y[i] * G

    g1p_fr2g1p_wrapper<<<32, 256>>>(Q, rows*512, X);

    CUDASYNC;

    for (int c=0; c<2; c++) {   // Tests must pass when c==0 and fail when c==1

        // IFT(FFT(P)) == P

        printf("=== RUN   IFT(FFT(P)) == P\n");
        for (i=0; i<512*512; i++) cmp[i] = 0;

        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(S, P); CUDASYNC;
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(T, S); CUDASYNC;
        if (c==1)
            g1p_gen(T[511]);

        g1p_eq_wrapper <<<rows, 256>>>(cmp, rows*512, P, T); CUDASYNC;

        for (i=0, pass=true; pass && (i<rows*512); i++)
            if (cmp[i] != 1) { fprintf(stderr, "ERROR at %d\n", i); pass = false; }

        PRINTPASS(pass^(c==1));

        // FFT(IFT(P)) == P

        printf("=== RUN   FFT(IFT(P)) == P\n");
        for (i=0; i<512*512; i++) cmp[i] = 0;

        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(T, P); CUDASYNC;
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(S, T); CUDASYNC;
        if (c==1)
            g1p_gen(S[511]);

        g1p_eq_wrapper <<<rows, 256>>>(cmp, rows*512, P, S); CUDASYNC;

        for (i=0, pass=true; pass && (i<rows*512); i++)
            if (cmp[i] != 1) { fprintf(stderr, "ERROR at %d\n", i); pass = false; }

        PRINTPASS(pass^(c==1));

        // FFT(P+Q) == FFT(P) + FFT(Q)

        printf("=== RUN   FFT(P+Q) == FFT(P) + FFT(Q)\n");
        for (i=0; i<512*512; i++) cmp[i] = 0;

        g1p_add_wrapper<<<rows, 256>>>(R, rows*512, P, Q); CUDASYNC;    // P+Q
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(R, R); CUDASYNC;  // FFT(P+Q)
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(S, P); CUDASYNC;  // FFT(P)
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(T, Q); CUDASYNC;  // FFT(Q)
        g1p_add_wrapper<<<rows, 256>>>(S, rows*512, S, T); CUDASYNC;    // FFT(P)+FFT(Q)
        if (c==1)
            g1p_gen(S[511]);

        g1p_eq_wrapper <<<rows, 256>>>(cmp, rows*512, R, S); CUDASYNC;

        for (i=0, pass=true; pass && (i<rows*512); i++)
            if (cmp[i] != 1) { fprintf(stderr, "ERROR at %d\n", i); pass = false; }

        PRINTPASS(pass^(c==1));

        // IFT(P+Q) == IFT(P) + IFT(Q)

        printf("=== RUN   IFT(P+Q) == IFT(P) + IFT(Q)\n");
        for (i=0; i<512*512; i++) cmp[i] = 0;

        g1p_add_wrapper<<<rows, 256>>>(R, rows*512, P, Q); CUDASYNC;    // P+Q
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(R, R); CUDASYNC;  // IFT(P+Q)
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(S, P); CUDASYNC;  // IFT(P)
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(T, Q); CUDASYNC;  // IFT(Q)
        g1p_add_wrapper<<<rows, 256>>>(S, rows*512, S, T); CUDASYNC;    // IFT(P)+IFT(Q)
        if (c==1)
            g1p_gen(S[511]);

        g1p_eq_wrapper <<<rows, 256>>>(cmp, rows*512, R, S); CUDASYNC;

        for (i=0, pass=true; pass && (i<rows*512); i++)
            if (cmp[i] != 1) { fprintf(stderr, "ERROR at %d\n", i); pass = false; }

        PRINTPASS(pass^(c==1));

        // FFT(x*P) == x*FFT(P)

        printf("=== RUN   FFT(x*P) == x*FFT(P)\n");
        for (i=0; i<512*512; i++) cmp[i] = 0;
        for (i=0; i<512*512; i++) fr_cpy(Z[i], Y[0]);

        g1p_mul_wrapper<<<rows, 256>>>(R, rows*512, Z, P); CUDASYNC;    // x*P
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(R, R); CUDASYNC;  // FFT(x*P)
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(S, P); CUDASYNC;  // FFT(P)
        g1p_mul_wrapper<<<rows, 256>>>(S, rows*512, Z, S); CUDASYNC;    // x*FFT(P)
        if (c==1)
            g1p_gen(S[511]);

        g1p_eq_wrapper <<<rows, 256>>>(cmp, rows*512, R, S); CUDASYNC;

        for (i=0, pass=true; pass && (i<rows*512); i++)
            if (cmp[i] != 1) { fprintf(stderr, "ERROR at %d\n", i); pass = false; }

        PRINTPASS(pass^(c==1));

        // IFT(x*P) == x*IFT(P)

        printf("=== RUN   IFT(x*P) == x*IFT(P)\n");
        for (i=0; i<512*512; i++) cmp[i] = 0;
        for (i=0; i<512*512; i++) fr_cpy(Z[i], Y[0]);

        g1p_mul_wrapper<<<rows, 256>>>(R, rows*512, Z, P); CUDASYNC;    // x*P
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(R, R); CUDASYNC;  // IFT(x*P)
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(S, P); CUDASYNC;  // IFT(P)
        g1p_mul_wrapper<<<rows, 256>>>(S, rows*512, Z, S); CUDASYNC;    // x*IFT(P)
        if (c==1)
            g1p_gen(S[511]);

        g1p_eq_wrapper <<<rows, 256>>>(cmp, rows*512, R, S); CUDASYNC;

        for (i=0, pass=true; pass && (i<rows*512); i++)
            if (cmp[i] != 1) { fprintf(stderr, "ERROR at %d\n", i); pass = false; }

        PRINTPASS(pass^(c==1));

        // FFT(G*X) == G*FFT(X) (FFT commutes with mapping from Fr to G1)

        printf("=== RUN   FFT(G*X) == G*FFT(X)\n");
        for (i=0; i<512*512; i++) cmp[i] = 0;
        for (i=0; i<512*512; i++) g1p_gen(R[i]);

        g1p_mul_wrapper<<<rows, 256>>>(S, rows*512, X, R); CUDASYNC;    // G*X
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(S, S); CUDASYNC;  // FFT(G*X)
        fr_fft_wrapper <<<rows, 256, fr_sharedmem>>> (Z, X); CUDASYNC;  // FFT(X)
        g1p_mul_wrapper<<<rows, 256>>>(T, rows*512, Z, R); CUDASYNC;    // G*FFT(X)
        if (c==1)
            g1p_gen(T[511]);

        g1p_eq_wrapper <<<rows, 256>>>(cmp, rows*512, S, T); CUDASYNC;

        for (i=0, pass=true; pass && (i<rows*512); i++)
            if (cmp[i] != 1) { fprintf(stderr, "ERROR at %d\n", i); pass = false; }

        PRINTPASS(pass^(c==1));

        // IFT(G*X) == G*IFT(X) (IFT commutes with mapping from Fr to G1)

        printf("=== RUN   IFT(G*X) == G*IFT(X)\n");
        for (i=0; i<512*512; i++) cmp[i] = 0;
        for (i=0; i<512*512; i++) g1p_gen(R[i]);

        g1p_mul_wrapper<<<rows, 256>>>(S, rows*512, X, R); CUDASYNC;    // G*X
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(S, S); CUDASYNC;  // IFT(G*X)
        fr_ift_wrapper <<<rows, 256, fr_sharedmem>>> (Z, X); CUDASYNC;  // IFT(X)
        g1p_mul_wrapper<<<rows, 256>>>(T, rows*512, Z, R); CUDASYNC;    // G*IFT(X)
        if (c==1)
            g1p_gen(T[511]);

        g1p_eq_wrapper <<<rows, 256>>>(cmp, rows*512, S, T); CUDASYNC;

        for (i=0, pass=true; pass && (i<rows*512); i++)
            if (cmp[i] != 1) { fprintf(stderr, "ERROR at %d\n", i); pass = false; }

        PRINTPASS(pass^(c==1));

        if (c==0) {
            printf("Tests below must detect an error at 511\n");
        }
    }

L_pf:
    fclose(pf);
}

// vim: ts=4 et sw=4 si