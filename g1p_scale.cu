// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "g1.cuh"

/**
 * @brief Scale the coordinates of a projective point.
 * This operation multiplies each coordinate of p by s: (x, y, z) := (x*s, y*s, z*s).
 * 
 * @param[in,out] p Point in G1 (stores result after call)
 * @param[in] s Multiplicand in Fp. Must be nonzero.
 * @return void
 */
__device__ void g1p_scale(g1p_t &p, const fp_t &s) {
    fp_mul(p.x, p.x, s);
    fp_mul(p.y, p.y, s);
    fp_mul(p.z, p.z, s);
}

// vim: ts=4 et sw=4 si
