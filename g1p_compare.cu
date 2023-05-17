// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fr.cuh"
#include "g1.cuh"

// g1p_eq compares one point to another, returns true when equal.
__device__ bool g1p_eq(const g1p_t &p, const g1p_t &q) {
    fp_t px, py, qx, qy;

    // (X1/Z1 == X2/Z2) && (Y1/Z1 == Y2/Z2)

    fp_cpy(px, p.x);
    fp_cpy(py, p.y);

    fp_cpy(qx, q.x);
    fp_cpy(qy, q.y);

    fp_mul(px, px, q.z);    // X1*Z2
    fp_mul(qx, qx, p.z);    // X2*Z1

    if (fp_neq(px, qx))
        return false;

    fp_mul(py, py, q.z);    // Y1*Z2
    fp_mul(qy, qy, p.z);    // Y2*Z1

    return fp_eq(py, qy);
}

// g1p_neq compares one point to another, returns true when not equal.
__device__ bool g1p_neq(const g1p_t &p, const g1p_t &q) {
    fp_t px, py, qx, qy;

    // (X1/Z1 == X2/Z2) && (Y1/Z1 == Y2/Z2)

    fp_cpy(px, p.x);
    fp_cpy(py, p.y);

    fp_cpy(qx, q.x);
    fp_cpy(qy, q.y);

    fp_mul(px, px, q.z);    // X1*Z2
    fp_mul(qx, qx, p.z);    // X2*Z1

    if (fp_neq(px, qx))
        return true;

    fp_mul(py, py, q.z);    // Y1*Z2
    fp_mul(qy, qy, p.z);    // Y2*Z1

    return fp_neq(py, qy);
}

// vim: ts=4 et sw=4 si
