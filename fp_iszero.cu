// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"

__device__ bool fp_iszero(const uint64_t *z) {
    return ((z[5] | z[4] | z[3] | z[2] | z[1] | z[0]) == 0);
}

// vim: ts=4 et sw=4 si
