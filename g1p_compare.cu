// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fp.cuh"
#include "fr.cuh"
#include "g1.cuh"

/**
 * @brief Compares two projective points, returns true when equal. This function
 * compares if both parameters represent the same point on the curve. The 
 * equality is given by comparing X and Y coordinates divided by Z coordinates
 * (p.X/p.Z == q.X/q.Z) && (p.Y/p.Z == q.Y/q.Z)
 * 
 * @param[in] p Projective G1 point
 * @param[in] q Projective G1 point
 * @return bool 1 if equal, 0 otherwise 
 */
__device__ bool g1p_eq(const g1p_t &p, const g1p_t &q) {
    fp_t px, py, qx, qy;

#ifndef NDEBUG
    if (!g1p_isPoint(p) || !(g1p_isPoint(q))) {
       // printf("ERROR in g1p_eq(): Invalid point(s)\n");
       // g1p_print("p:", p);
       // g1p_print("q:", q);

        return false;
    }
#endif

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

/**
 * @brief Compares two projective points, returns true when not equal. This function
 * compares if both parameters represent the distinct points on the curve. The 
 * equality is given by comparing X and Y coordinates divided by Z coordinates
 * (p.X/p.Z == q.X/q.Z) && (p.Y/p.Z == q.Y/q.Z)
 * 
 * @param[in] p Projective G1 point
 * @param[in] q Projective G1 point
 * @return bool 0 if equal, 1 otherwise 
 */
__device__ bool g1p_neq(const g1p_t &p, const g1p_t &q) {
    fp_t px, py, qx, qy;

#ifndef NDEBUG
    if (!g1p_isPoint(p) || !(g1p_isPoint(q))) {
        printf("ERROR in g1p_neq(): Invalid point(s)\n");
        g1p_print("p:", p);
        g1p_print("q:", q);

        return true;
    }
#endif

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
