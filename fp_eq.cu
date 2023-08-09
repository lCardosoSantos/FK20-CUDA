// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"

/**
 * @brief Compares two fp_t residues
 * 
 * @param[in] x 
 * @param[in] y 
 * @return bool 1 if the values are equal, 0 otherwise
 */
__device__ bool fp_eq(const fp_t &x, const fp_t &y) {
        fp_t t;

        fp_sub(t, x, y);

        return fp_iszero(t);
}

// vim: ts=4 et sw=4 si
