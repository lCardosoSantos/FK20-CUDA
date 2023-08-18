// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"

/**
 * @brief Copy from x into z.
 *
 * @param[out] z
 * @param[in] x
 * @return  void
 */
__device__ __host__ void fr_cpy(fr_t &z, const fr_t &x) {
        z[0] = x[0];
        z[1] = x[1];
        z[2] = x[2];
        z[3] = x[3];
}

// vim: ts=4 et sw=4 si
