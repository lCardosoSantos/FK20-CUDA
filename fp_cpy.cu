// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"

/**
 * @brief Copy from x into z.
 *
 * @param[out] z
 * @param[in] x
 * @return void
 */
__device__ __host__ void fp_cpy(fp_t &z, const fp_t &x) {
    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
    z[4] = x[4];
    z[5] = x[5];
}

// vim: ts=4 et sw=4 si
