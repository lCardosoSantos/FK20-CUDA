// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "g1.cuh"

__device__ void g1p_neg(uint64_t *p) {
    if (fp_nonzero(p+12))
        fp_neg(p+6, p+6);
}

// vim: ts=4 et sw=4 si
