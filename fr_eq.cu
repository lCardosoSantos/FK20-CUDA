// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
/**
 * @brief Compares two residues modulo r
 * 
 * @param[in] x 
 * @param[in] y 
 * @return bool 1 if the values are equal, 0 otherwise
 */
__device__ bool fr_eq(const fr_t &x, const fr_t &y) {
        fr_t t;

        fr_cpy(t, x);
        fr_sub(t, y);

        return fr_iszero(t); //returns one if equal
}

// vim: ts=4 et sw=4 si
