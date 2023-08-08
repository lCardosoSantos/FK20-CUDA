// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos


#include "fr.cuh"

/**
 * @brief Returns 1 if the operands are not equal. 
 * 
 * @param[in] x 
 * @param[in] y 
 * @return bool 
 */
__device__ bool fr_neq(const fr_t &x, const fr_t &y) {
        return !fr_eq(x, y);
}

// vim: ts=4 et sw=4 si
