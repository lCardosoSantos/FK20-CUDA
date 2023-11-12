// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_SQR_CUH
#define FP_SQR_CUH

#include <cstdio>

#include "fp.cuh"
#include "fp_mul.cuh"
#include "ptx.cuh"

__device__ __forceinline__ void fp_sqr(
    uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3, uint64_t &z4, uint64_t &z5,
    uint64_t &z6, uint64_t &z7, uint64_t &z8, uint64_t &z9, uint64_t &za, uint64_t &zb,
    uint64_t  x0, uint64_t  x1, uint64_t  x2, uint64_t  x3, uint64_t  x4, uint64_t  x5
    )
{
    uint32_t
        xl0, xl1, xl2, xl3, xl4, xl5,
        xh0, xh1, xh2, xh3, xh4, xh5,
        l0, l1, h0, h1;

    uint64_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb;


    unpack(xl0, xh0, x0);
    unpack(xl1, xh1, x1);
    unpack(xl2, xh2, x2);
    unpack(xl3, xh3, x3);
    unpack(xl4, xh4, x4);
    unpack(xl5, xh5, x5);


    mul_wide_u32(t0, xl0, xh0);
    mul_wide_u32(t1, xl0, xh1);
    mul_wide_u32(t2, xl0, xh2);
    mul_wide_u32(t3, xl0, xh3);
    mul_wide_u32(t4, xl0, xh4);
    mul_wide_u32(t5, xl0, xh5);

    mul_wide_u32(z1, xl0, xl1);
    mul_wide_u32(z2, xl0, xl2);
    mul_wide_u32(z3, xl0, xl3);
    mul_wide_u32(z4, xl0, xl4);
    mul_wide_u32(z5, xl0, xl5);


    mad_wide_cc_u32 (t1, xh0, xl1, t1);
    madc_wide_cc_u32(t2, xh0, xl2, t2);
    madc_wide_cc_u32(t3, xh0, xl3, t3);
    madc_wide_cc_u32(t4, xh0, xl4, t4);
    madc_wide_cc_u32(t5, xh0, xl5, t5);
    addc_u64        (t6,   0,   0);

    mad_wide_cc_u32 (z2, xh0, xh1, z2);
    madc_wide_cc_u32(z3, xh0, xh2, z3);
    madc_wide_cc_u32(z4, xh0, xh3, z4);
    madc_wide_cc_u32(z5, xh0, xh4, z5);
    madc_wide_u32   (z6, xh0, xh5,  0);


    mad_wide_cc_u32 (t2, xl1, xh1, t2);
    madc_wide_cc_u32(t3, xl1, xh2, t3);
    madc_wide_cc_u32(t4, xl1, xh3, t4);
    madc_wide_cc_u32(t5, xl1, xh4, t5);
    madc_wide_u32   (t6, xl1, xh5, t6);

    mad_wide_cc_u32 (z3, xl1, xl2, z3);
    madc_wide_cc_u32(z4, xl1, xl3, z4);
    madc_wide_cc_u32(z5, xl1, xl4, z5);
    madc_wide_cc_u32(z6, xl1, xl5, z6);
    addc_u64        (z7,   0,   0);


    mad_wide_cc_u32 (t3, xh1, xl2, t3);
    madc_wide_cc_u32(t4, xh1, xl3, t4);
    madc_wide_cc_u32(t5, xh1, xl4, t5);
    madc_wide_cc_u32(t6, xh1, xl5, t6);
    addc_u64        (t7,   0,   0);

    mad_wide_cc_u32 (z4, xh1, xh2, z4);
    madc_wide_cc_u32(z5, xh1, xh3, z5);
    madc_wide_cc_u32(z6, xh1, xh4, z6);
    madc_wide_u32   (z7, xh1, xh5, z7);


    mad_wide_cc_u32 (t4, xl2, xh2, t4);
    madc_wide_cc_u32(t5, xl2, xh3, t5);
    madc_wide_cc_u32(t6, xl2, xh4, t6);
    madc_wide_u32   (t7, xl2, xh5, t7);

    mad_wide_cc_u32 (z5, xl2, xl3, z5);
    madc_wide_cc_u32(z6, xl2, xl4, z6);
    madc_wide_cc_u32(z7, xl2, xl5, z7);
    addc_u64        (z8,   0,   0);


    mad_wide_cc_u32 (t5, xh2, xl3, t5);
    madc_wide_cc_u32(t6, xh2, xl4, t6);
    madc_wide_cc_u32(t7, xh2, xl5, t7);
    addc_u64        (t8,   0,   0);

    mad_wide_cc_u32 (z6, xh2, xh3, z6);
    madc_wide_cc_u32(z7, xh2, xh4, z7);
    madc_wide_u32   (z8, xh2, xh5, z8);


    mad_wide_cc_u32 (t6, xl3, xh3, t6);
    madc_wide_cc_u32(t7, xl3, xh4, t7);
    madc_wide_u32   (t8, xl3, xh5, t8);

    mad_wide_cc_u32 (z7, xl3, xl4, z7);
    madc_wide_cc_u32(z8, xl3, xl5, z8);
    addc_u64        (z9,   0,   0);


    mad_wide_cc_u32 (t7, xh3, xl4, t7);
    madc_wide_cc_u32(t8, xh3, xl5, t8);
    addc_u64        (t9,   0,   0);

    mad_wide_cc_u32 (z8, xh3, xh4, z8);
    madc_wide_u32   (z9, xh3, xh5, z9);


    mad_wide_cc_u32 (t8, xl4, xh4, t8);
    madc_wide_u32   (t9, xl4, xh5, t9);

    mad_wide_cc_u32 (z9, xl4, xl5, z9);
    madc_wide_u32   (za, xh4, xh5,  0);


    mad_wide_cc_u32 (t9, xh4, xl5, t9);
    madc_wide_u32   (ta, xl5, xh5,  0);

    // t += z >> 32

    unpack(l0, h0, t0);                           unpack(l1, h1, z1);  add_cc_u32 (h0, h0, l1); pack(t0, l0, h0);
    unpack(l0, h0, t1);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, z2);  addc_cc_u32(h0, h0, l1); pack(t1, l0, h0);
    unpack(l0, h0, t2);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, z3);  addc_cc_u32(h0, h0, l1); pack(t2, l0, h0);
    unpack(l0, h0, t3);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, z4);  addc_cc_u32(h0, h0, l1); pack(t3, l0, h0);
    unpack(l0, h0, t4);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, z5);  addc_cc_u32(h0, h0, l1); pack(t4, l0, h0);
    unpack(l0, h0, t5);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, z6);  addc_cc_u32(h0, h0, l1); pack(t5, l0, h0);
    unpack(l0, h0, t6);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, z7);  addc_cc_u32(h0, h0, l1); pack(t6, l0, h0);
    unpack(l0, h0, t7);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, z8);  addc_cc_u32(h0, h0, l1); pack(t7, l0, h0);
    unpack(l0, h0, t8);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, z9);  addc_cc_u32(h0, h0, l1); pack(t8, l0, h0);
    unpack(l0, h0, t9);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, za);  addc_cc_u32(h0, h0, l1); pack(t9, l0, h0);
    unpack(l0, h0, ta);  addc_cc_u32(l0, l0, h1); unpack(l1, h1,  0);  addc_u32   (h0, h0,  0); pack(ta, l0, h0);

    // t += t

    add_cc_u64 (t0, t0, t0);
    addc_cc_u64(t1, t1, t1);
    addc_cc_u64(t2, t2, t2);
    addc_cc_u64(t3, t3, t3);
    addc_cc_u64(t4, t4, t4);
    addc_cc_u64(t5, t5, t5);
    addc_cc_u64(t6, t6, t6);
    addc_cc_u64(t7, t7, t7);
    addc_cc_u64(t8, t8, t8);
    addc_cc_u64(t9, t9, t9);
    addc_cc_u64(ta, ta, ta);
    addc_u64   (tb,  0,  0);

    // Store squares in z

    mul_wide_u32(z0, xl0, xl0); mul_wide_u32(z1, xh0, xh0);
    mul_wide_u32(z2, xl1, xl1); mul_wide_u32(z3, xh1, xh1);
    mul_wide_u32(z4, xl2, xl2); mul_wide_u32(z5, xh2, xh2);
    mul_wide_u32(z6, xl3, xl3); mul_wide_u32(z7, xh3, xh3);
    mul_wide_u32(z8, xl4, xl4); mul_wide_u32(z9, xh4, xh4);
    mul_wide_u32(za, xl5, xl5); mul_wide_u32(zb, xh5, xh5);

    // z += t >> 32

    unpack(l0, h0, z0);                           unpack(l1, h1, t0);  add_cc_u32 (h0, h0, l1); pack(z0, l0, h0);
    unpack(l0, h0, z1);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t1);  addc_cc_u32(h0, h0, l1); pack(z1, l0, h0);
    unpack(l0, h0, z2);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t2);  addc_cc_u32(h0, h0, l1); pack(z2, l0, h0);
    unpack(l0, h0, z3);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t3);  addc_cc_u32(h0, h0, l1); pack(z3, l0, h0);
    unpack(l0, h0, z4);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t4);  addc_cc_u32(h0, h0, l1); pack(z4, l0, h0);
    unpack(l0, h0, z5);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t5);  addc_cc_u32(h0, h0, l1); pack(z5, l0, h0);
    unpack(l0, h0, z6);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t6);  addc_cc_u32(h0, h0, l1); pack(z6, l0, h0);
    unpack(l0, h0, z7);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t7);  addc_cc_u32(h0, h0, l1); pack(z7, l0, h0);
    unpack(l0, h0, z8);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t8);  addc_cc_u32(h0, h0, l1); pack(z8, l0, h0);
    unpack(l0, h0, z9);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, t9);  addc_cc_u32(h0, h0, l1); pack(z9, l0, h0);
    unpack(l0, h0, za);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, ta);  addc_cc_u32(h0, h0, l1); pack(za, l0, h0);
    unpack(l0, h0, zb);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, tb);  addc_u32   (h0, h0, l1); pack(zb, l0, h0);
}

#endif
// vim: ts=4 et sw=4 si
