// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_X8

#include "ptx.cuh"
#include "fp_reduce7.cuh"

__forceinline__
__device__ void fp_x8(
    uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3, uint64_t &z4, uint64_t &z5, uint64_t &z6,
    uint64_t  x0, uint64_t  x1, uint64_t  x2, uint64_t  x3, uint64_t  x4, uint64_t  x5
    )
{
    uint64_t x6;

    // x = x + x

    add_cc_u64 (x0, x0, x0);
    addc_cc_u64(x1, x1, x1);
    addc_cc_u64(x2, x2, x2);
    addc_cc_u64(x3, x3, x3);
    addc_cc_u64(x4, x4, x4);
    addc_cc_u64(x5, x5, x5);
    addc_cc_u64(x6,  0,  0);

    // x = x + x

    add_cc_u64 (x0, x0, x0);
    addc_cc_u64(x1, x1, x1);
    addc_cc_u64(x2, x2, x2);
    addc_cc_u64(x3, x3, x3);
    addc_cc_u64(x4, x4, x4);
    addc_cc_u64(x5, x5, x5);
    addc_cc_u64(x6, x6, x6);

    // z = x + x

    add_cc_u64 (z0, x0, x0);
    addc_cc_u64(z1, x1, x1);
    addc_cc_u64(z2, x2, x2);
    addc_cc_u64(z3, x3, x3);
    addc_cc_u64(z4, x4, x4);
    addc_cc_u64(z5, x5, x5);
    addc_cc_u64(z6, x6, x6);
}

__forceinline__
__device__ void fp_x8(
    uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3, uint64_t &z4, uint64_t &z5,
    uint64_t  x0, uint64_t  x1, uint64_t  x2, uint64_t  x3, uint64_t  x4, uint64_t  x5)
{
    uint64_t t0, t1, t2, t3, t4, t5, t6;

    fp_x8(
        t0, t1, t2, t3, t4, t5, t6,
        x0, x1, x2, x3, x4, x5
    );

    fp_reduce7(
        z0, z1, z2, z3, z4, z5,
        t0, t1, t2, t3, t4, t5, t6
    );
}

#endif
// vim: ts=4 et sw=4 si
