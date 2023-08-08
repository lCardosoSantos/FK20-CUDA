// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"

__device__ bool fp_neq(const fp_t &x, const fp_t &y) {
        return !fp_eq(x,y);
}

// vim: ts=4 et sw=4 si
