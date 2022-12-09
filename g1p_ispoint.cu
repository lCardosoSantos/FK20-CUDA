// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "g1.cuh"

__device__ bool g1p_isInf(const uint64_t *p) {
    return fp_iszero(p+0) && fp_isone(p+6) && fp_iszero(p+12);
}

__device__ bool g1p_isPoint(const uint64_t *p) {
    if (g1p_isInf(p))
        return true;

    if (fp_iszero(p+12))
        return false;

    uint64_t x[6], y[6], z[6];

    fp_cpy(x, p+ 0);
    fp_cpy(y, p+ 6);
    fp_cpy(z, p+12);

    fp_sqr(y, y);       // Y^2
    fp_mul(y, y, z);    // Y^2*Z

    fp_sqr(x, x);       // X^2
    fp_mul(x, x, p);    // X^3

    fp_sqr(z, z);       // Z^2
    fp_mul(z, z, p+12); // Z^3
    fp_x4(z, z);

    fp_add(x, x, z);    // X^3 + 4*Z^3

    return fp_eq(x, y);
}

// vim: ts=4 et sw=4 si
