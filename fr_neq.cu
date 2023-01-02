// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"

__device__ bool fr_neq(const fr_t &x, const fr_t &y) {
        fr_t t;

        fr_cpy(t, x);
        fr_sub(t, y);

        return fr_nonzero(t);
}

// vim: ts=4 et sw=4 si
