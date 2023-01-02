// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"

__device__ bool fp_eq(const fp_t &x, const fp_t &y) {
        fp_t t;

        fp_sub(t, x, y);

        return fp_iszero(t);
}

// vim: ts=4 et sw=4 si
