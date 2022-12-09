// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fr.cuh"
#include "g1.cuh"

// g1p_equal compares one point to another, returns true when equal.
__device__ bool g1p_equal(uint64_t *p, uint64_t *q) {
    uint64_t px[6], py[6], qx[6], qy[6];

    // (X1/Z1 == X2/Z2) && (Y1/Z1 == Y2/Z2)

    fp_cpy(px, p+0);
    fp_cpy(py, p+6);

    fp_cpy(qx, q+0);
    fp_cpy(qy, q+6);

    fp_mul(px, px, q+12);   // X1*Z2
    fp_mul(qx, qx, p+12);   // X2*Z1

    if (!fp_eq(px, qx))
        return false;

    fp_mul(py, py, q+12);   // Y1*Z2
    fp_mul(qy, qy, p+12);   // Y2*Z1

    return fp_eq(py, qy);
}

// g1p_neq compares one point to another, returns true when not equal.
__device__ bool g1p_neq(uint64_t *p, uint64_t *q) {
    uint64_t px[6], py[6], qx[6], qy[6];

    // (X1/Z1 == X2/Z2) && (Y1/Z1 == Y2/Z2)

    fp_cpy(px, p+0);
    fp_cpy(py, p+6);

    fp_cpy(qx, q+0);
    fp_cpy(qy, q+6);

    fp_mul(px, px, q+12);	// X1*Z2
    fp_mul(qx, qx, p+12);	// X2*Z1

    if (fp_neq(px, qx))
        return true;

    fp_mul(py, py, q+12);	// Y1*Z2
    fp_mul(qy, qy, p+12);	// Y2*Z1

    return fp_neq(py, qy);
}

// vim: ts=4 et sw=4 si
