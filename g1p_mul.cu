// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fr.cuh"
#include "g1.cuh"

// p ← k·p
__device__ void g1p_mul(g1p_t &p, const fr_t &k) {
    // TODO: Use 4-bit lookup table to reduce additions by a factor 4.

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
