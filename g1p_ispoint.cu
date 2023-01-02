// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "g1.cuh"

__device__ bool g1p_isInf(const g1p_t &p) {
    return fp_iszero(p.x) && fp_isone(p.y) && fp_iszero(p.z);
}

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
