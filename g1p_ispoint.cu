// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "g1.cuh"
/**
 * @brief Check if the value stored in p is the the/any point at infinity.
 * This implementation uses (0, 1, 0) as the point at infinity.
 * Alternatively, the macro G1P_ANYINF allows the point at infinity to be 
 * represented as (0, y, 0) where y!=0 
 * 
 * The algebra used in this library sets the point at infinity to (0, 1, 0) (or
 * (0, y, 0)), instead of the usual (1, 1, 0) used in other libs, due to following
 * the Algorithm 7 in eprint 2015-1060.
 * 
 * @param[in] p 
 * @return bool 1 if the p is the point at infinity.
 */
__device__ bool g1p_isInf(const g1p_t &p) {
#if G1P_ANYINF
    return fp_iszero(p.x) && !fp_iszero(p.y) && fp_iszero(p.z);
#else
    return fp_iszero(p.x) &&   fp_isone(p.y) && fp_iszero(p.z);
#endif
}

/**
 * @brief Check if the value stored in p is a valid point on the G1 curve.
 * 
 * @param[in] p 
 * @return bool 1 if is in the curve, zero otherwise. 
 */
__device__ bool g1p_isPoint(const g1p_t &p) {
    if (g1p_isInf(p))
        return true;

    if (fp_iszero(p.z))
        return false;

    fp_t x, y, z;

    fp_cpy(x, p.x);
    fp_cpy(y, p.y);
    fp_cpy(z, p.z);

    fp_sqr(y, y);       // Y^2
    fp_mul(y, y, z);    // Y^2*Z

    fp_sqr(x, x);       // X^2
    fp_mul(x, x, p.x);  // X^3

    fp_sqr(z, z);       // Z^2
    fp_mul(z, z, p.z);  // Z^3
    fp_x4(z, z);

    fp_add(x, x, z);    // X^3 + 4*Z^3

    return fp_eq(x, y);
}

// vim: ts=4 et sw=4 si
