// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "g1.cuh"
#include "fr.cuh"
#include "g1test.cuh"

/**
 * @brief Test addition and multiplication using a fibonacci sequence (cascading
 * data dependency)
 * 
 * @return void 
 */
__global__ void G1TestFibonacci(testval_t *) {

    if ((blockIdx.x | blockIdx.y | blockIdx.z | threadIdx.x | threadIdx.y | threadIdx.z) == 0)
        printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    g1p_t p, q, t;
    fr_t k, l;

    g1p_inf(p); // p  = 0
    g1p_gen(q); // q  = G

    fr_zero(k);
    fr_one(l);

    for (int i=0; pass && i<100; i++) {

        fr_add(k, l);
        g1p_add(p, q);  // p += q

        g1p_gen(t);
        g1p_mul(t, k);  // kG

        if (g1p_neq(p, t)) {
            pass = false;
        }
        ++count;

        if (!pass)
            break;

        fr_add(l, k);
        g1p_add(q, p);  // q += p

        g1p_gen(t);
        g1p_mul(t, l);  // lG

        if (g1p_neq(q, t)) {
            pass = false;
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
