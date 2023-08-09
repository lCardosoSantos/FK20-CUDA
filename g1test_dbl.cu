// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "g1.cuh"
#include "fr.cuh"
#include "g1test.cuh"

__global__ void G1TestDbl(testval_t *) {

    if ((blockIdx.x | blockIdx.y | blockIdx.z | threadIdx.x | threadIdx.y | threadIdx.z) == 0)
        printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    g1p_t p, q, u, v;

    g1p_gen(p); // p  = G
    g1p_gen(q); // q  = G

    for (int i=0; pass && i<20000; i++) {

        g1p_cpy(u, p);
        g1p_cpy(v, q);

        g1p_add(p, p);  // p += p
        g1p_dbl(q);     // q *= 2

        if (g1p_neq(p, q)) {
            pass = false;

            printf("%d: FAILED\n", i);
            g1p_print("u   = ", u);
            g1p_print("v   = ", v);
            g1p_print("u+u = ", p);
            g1p_print("2v  = ", q);
        }
        ++count;
    }

    if (!pass || (blockIdx.x | blockIdx.y | blockIdx.z | threadIdx.x | threadIdx.y | threadIdx.z) == 0)
    {
        printf("%ld tests\n", count);

        PRINTPASS(pass);
    }
}

// vim: ts=4 et sw=4 si
