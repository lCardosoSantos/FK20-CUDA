// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "g1.cuh"

__device__ void g1p_scale(g1p_t &p, const fp_t &s) {
    fp_mul(p.x, p.x, s);
    fp_mul(p.y, p.y, s);
    fp_mul(p.z, p.z, s);
}

// vim: ts=4 et sw=4 si
