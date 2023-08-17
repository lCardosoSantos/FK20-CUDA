// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"

/**
 * @brief Returns true iff x==0
 *
 * @param[in] x
 * @return bool 1 if true, 0 otherwise
 */
__device__ bool fp_iszero(const fp_t &x) {
    fp_t t;
    fp_cpy(t, x);
    fp_reduce6(t);

    return (t[5] | t[4] | t[3] | t[2] | t[1] | t[0]) == 0;
}

// vim: ts=4 et sw=4 si
