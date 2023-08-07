// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "g1.cuh"

/**
 * @brief Elliptic curve scaling. p = p*s
 * This operation multiplies each point coordinate of p by s.
 * 
 * @param[in,out] p point in G1 (stores result after call)
 * @param[in] s multiplicand in Fp
 * @return void
 */
__device__ void g1p_scale(g1p_t &p, const fp_t &s) {
    fp_mul(p.x, p.x, s);
    fp_mul(p.y, p.y, s);
    fp_mul(p.z, p.z, s);
}

// vim: ts=4 et sw=4 si
