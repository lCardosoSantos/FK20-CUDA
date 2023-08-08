// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "g1.cuh"

/**
 * @brief Computes the negative of the point p. 
 * Due to negation map automorphism on Elliptic Curves in Weierstrass form, this
 * operation is done by computing the additive inverse of the Y coordinate.
 * 
 * @param[in,out] p 
 * @return void 
 */
__device__ void g1p_neg(g1p_t &p) {
    if (fp_nonzero(p.z))
        fp_neg(p.y, p.y);
}

// vim: ts=4 et sw=4 si
