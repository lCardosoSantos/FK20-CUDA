// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"

__device__ bool fr_iszero(const fr_t &x) {
    fr_t t;
    fr_cpy(t, x);
    fr_reduce4(t);

    return ((t[3] | t[2] | t[1] | t[0]) == 0);
}

// vim: ts=4 et sw=4 si
