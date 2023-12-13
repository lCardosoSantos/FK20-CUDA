// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "g1.cuh"
#include "fp.cuh"
#include "fp_cpy.cuh"
#include "fp_x2.cuh"
#include "fp_x3.cuh"
#include "fp_x4.cuh"
#include "fp_x8.cuh"
#include "fp_x12.cuh"
#include "fp_add.cuh"
#include "fp_sub.cuh"
#include "fp_sqr.cuh"
#include "fp_mul.cuh"
#include "fp_reduce12.cuh"

/**
 * @brief G1 point doubling. p ← 2*p
 * previous implementation, without unrolling of the PTX to in-register.
 * 
 * @param[in, out] p 
 * @return void 
 */
__device__ void g1p_dbl_noPTX(g1p_t &p) {

    if (!g1p_isPoint(p)) {
        g1p_print("ERROR in g1p_dbl(): Invalid point ", p);

        // return invalid point as result
        fp_zero(p.x);
        fp_zero(p.y);
        fp_zero(p.z);

        return;
    } 

    fp_t x, y, z, v, w;

    fp_cpy(x, p.x);
    fp_cpy(y, p.y);
    fp_cpy(z, p.z);

    fp_mul(x, x, y);
    fp_sqr(v, z);
    fp_x12(v, v);
    fp_mul(z, z, y);
    fp_sqr(y, y);
    fp_x3(w, v);
    fp_sub(w, y, w);
    fp_mul(x, x, w);
    fp_add(y, y, v);
    fp_mul(w, w, y);
    fp_sub(y, y, v);
    fp_x8(y, y);
    fp_x2(x, x);
    fp_mul(z, z, y);
    fp_mul(y, y, v);
    fp_add(y, y, w);

    fp_cpy(p.x, x);
    fp_cpy(p.y, y);
    fp_cpy(p.z, z);

}

// vim: ts=4 et sw=4 si
