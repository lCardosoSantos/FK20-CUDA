// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fp_x8.cuh"
#include "fp_reduce7.cuh"

/**
 * @brief Multiplies x by 8 and stores the result into z.
 *
 * @param[out] z
 * @param[in] x
 * @return void
 */
__device__ void fp_x8(fp_t &z, const fp_t &x) {

    fp_x8(
        z[0], z[1], z[2], z[3], z[4], z[5],
        x[0], x[1], x[2], x[3], x[4], x[5]
    );
}

// vim: ts=4 et sw=4 si
