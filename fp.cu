// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "fp.cuh"

__device__ __host__ void fp_zero(fp_t &z) {
    for (int i=0; i<6; i++)
        z[i] = 0;
}

__device__ __host__ void fp_one(fp_t &z) {
    z[0] = 1;
    for (int i=1; i<6; i++)
        z[i] = 0;
}

__device__ void fp_print(const fp_t &x) {
    fp_t t;
    fp_cpy(t, x);
    fp_reduce6(t);
//  printf("#x%016lx%016lx%016lx%016lx%016lx%016lx\n",  // clisp
    printf("%016lX%016lX%016lX%016lX%016lX%016lX\n",    // dc
//  printf("0x%016lx%016lx%016lx%016lx%016lx%016lx\n",  // python
    t[5], t[4], t[3], t[2], t[1], t[0]);
}

__device__ __host__ void fp_fromUint64(fp_t &z, const uint64_t *x) {
    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
    z[4] = x[4];
    z[5] = x[5];
}

__device__ void fp_toUint64(const fp_t &x, uint64_t *z) {
    fp_t t;
    fp_cpy(t, x);
    fp_reduce6(t);

    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
    z[4] = x[4];
    z[5] = x[5];
}

// vim: ts=4 et sw=4 si
