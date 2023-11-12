// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fp_sqr.cuh"
#include "fp_mul.cuh"
#include "fp_reduce12.cuh"

/**
 * @brief Computes the square of the residue x modulo p and stores it in z.
 *
 * @param[out] z
 * @param[in] x
 * @return void
 */
__device__ void fp_sqr(fp_t &z, const fp_t &x) {

    uint64_t
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb,
        x0, x1, x2, x3, x4, x5,
        z0, z1, z2, z3, z4, z5;

    x0 = x[0];
    x1 = x[1];
    x2 = x[2];
    x3 = x[3];
    x4 = x[4];
    x5 = x[5];

    fp_sqr(
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb,
        x0, x1, x2, x3, x4, x5
    );

    fp_reduce12(
        z0, z1, z2, z3, z4, z5,
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb
    );

    z[0x0] = z0;
    z[0x1] = z1;
    z[0x2] = z2;
    z[0x3] = z3;
    z[0x4] = z4;
    z[0x5] = z5;
}

// vim: ts=4 et sw=4 si
