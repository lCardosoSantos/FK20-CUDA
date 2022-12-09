// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "g1.cuh"

__device__ void g1p_scale(uint64_t *p, const uint64_t *s) {
    fp_mul(p+ 0, p+ 0, s);
    fp_mul(p+ 6, p+ 6, s);
    fp_mul(p+12, p+12, s);
}

// vim: ts=4 et sw=4 si
