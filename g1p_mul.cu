// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fp.cuh"
#include "fr.cuh"
#include "g1.cuh"

/**
 * @brief Point multiplication by scalar, in projective coordinates. p ← k·p
 * 
 * @param[in, out] p Multiplicand (stores result after call)
 * @param[in] k Fr multiplier
 * @return void 
 */
__device__ void g1p_mul(g1p_t &p, const fr_t &k) {
    // TODO: Use 4-bit lookup table to reduce additions by a factor 4.
    
#if 0 //ndef NDEBUG
    if (!g1p_isPoint(p)) {
        //g1p_print("ERROR in g1p_mul(): Invalid point ", p);

        // return invalid point as result
        fp_zero(p.x);
        fp_zero(p.y);
        fp_zero(p.z);

        return;
    }
#endif

    g1p_t q;

    g1p_inf(q); // q = inf

    for (int i=3; i>=0; i--) {
        uint64_t
            t = k[i],
            j = 1ULL<<63;

        for (; j!=0; j>>=1) {
            g1p_dbl(q);

            if ((t&j) != 0) {
                g1p_add(q, p);
            }
        }
    }

    g1p_cpy(p, q);
}

// vim: ts=4 et sw=4 si
