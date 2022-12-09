// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "fr.cuh"

__device__ void fr_zero(uint64_t *z) {
    for (int i=0; i<4; i++)
        z[i] = 0;
}

__device__ void fr_one(uint64_t *z) {
    z[0] = 1;
    for (int i=1; i<4; i++)
        z[i] = 0;
}

__device__ void fr_print(const uint64_t *x) {
    printf("%016lx%016lx%016lx%016lx\n",
    x[3], x[2], x[1], x[0]);
}

// vim: ts=4 et sw=4 si
