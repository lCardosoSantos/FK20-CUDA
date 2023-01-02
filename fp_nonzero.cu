// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"

__device__ bool fp_nonzero(const fp_t &x) {
    fp_t t;
    fp_cpy(t, x);
    fp_reduce6(t);

    return (t[5] | t[4] | t[3] | t[2] | t[1] | t[0]) != 0;
}

// vim: ts=4 et sw=4 si
