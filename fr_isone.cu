// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"

/**
 * @brief checks if the reduced scalar x is equal to one
 * 
 * @param[in] x 
 * @return bool 1 if the reduced value is one, 0 otherwise 
 */
__device__ bool fr_isone(const fr_t &x) {
    fr_t t;
    fr_cpy(t, x);
    fr_reduce4(t);

    return ((t[3] | t[2] | t[1] | (1 ^ t[0])) == 0);
}

// vim: ts=4 et sw=4 si
