// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"

__device__ bool fr_neq(uint64_t *x, uint64_t *y) {
        uint64_t t;

        fr_reduce4(x);
        fr_reduce4(y);

        t  = x[0] ^ y[0];
        t |= x[1] ^ y[1];
        t |= x[2] ^ y[2];
        t |= x[3] ^ y[3];
        return t != 0;
}

// vim: ts=4 et sw=4 si
