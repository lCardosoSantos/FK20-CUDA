// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"

__device__ void fp_cpy(fp_t &z, const fp_t &x) {
    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
    z[4] = x[4];
    z[5] = x[5];
}

// vim: ts=4 et sw=4 si
