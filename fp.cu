// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "fp.cuh"

__device__ void fp_zero(uint64_t *z) {
    for (int i=0; i<6; i++)
        z[i] = 0;
}

__device__ void fp_one(uint64_t *z) {
    z[0] = 1;
    for (int i=1; i<6; i++)
        z[i] = 0;
}

__device__ void fp_print(const uint64_t *x) {
    uint64_t t[6];
    fp_cpy(t, x);
    fp_reduce6(t);
    printf("0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
    t[5], t[4], t[3], t[2], t[1], t[0]);
}

// vim: ts=4 et sw=4 si
