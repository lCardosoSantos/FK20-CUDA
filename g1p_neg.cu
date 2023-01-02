// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "g1.cuh"

__device__ void g1p_neg(g1p_t &p) {
    if (fp_nonzero(p.z))
        fp_neg(p.y, p.y);
}

// vim: ts=4 et sw=4 si
