// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "fr.cuh"

__device__ __host__ void fr_zero(fr_t &z) {
    for (int i=0; i<4; i++)
        z[i] = 0;
}

__device__ __host__ void fr_one(fr_t &z) {
    z[0] = 1;
    for (int i=1; i<4; i++)
        z[i] = 0;
}

__device__ void fr_print(const fr_t &x) {
    fr_t t;
    fr_cpy(t, x);
    fr_reduce4(t);
//  printf("#x%016lx%016lx%016lx%016lx\n",  // clisp
    printf("%016lX%016lX%016lX%016lX\n",    // dc
//  printf("0x%016lx%016lx%016lx%016lx\n",  // python
    t[3], t[2], t[1], t[0]);
}

__device__ __host__ void fr_fromUint64(fr_t &z, const uint64_t *x) {
    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
}

__device__ void fr_toUint64(const fr_t &x, uint64_t *z) {
    fr_t t;
    fr_cpy(t, x);
    fr_reduce4(t);

    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
}

////////////////////////////////////////////////////////////

__global__ void fr_eq_wrapper(uint8_t *eq, int count, const fr_t *x, const fr_t *y) {

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    unsigned step = gridDim.z * gridDim.y * gridDim.x
                * blockDim.z * blockDim.y * blockDim.x;

    for (unsigned i=tid; i<count; i+=step)
        eq[i] = fr_eq(x[i], y[i]) ? 1 : 0;
}

// vim: ts=4 et sw=4 si
