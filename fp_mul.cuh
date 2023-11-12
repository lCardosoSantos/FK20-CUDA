// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_MUL_CUH
#define FP_MUL_CUH

#include <cstdio>

#include "fp.cuh"
#include "ptx.cuh"

__device__ __forceinline__ void fp_mul(
    uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3, uint64_t &z4, uint64_t &z5,
    uint64_t &z6, uint64_t &z7, uint64_t &z8, uint64_t &z9, uint64_t &za, uint64_t &zb,
    uint64_t  x0, uint64_t  x1, uint64_t  x2, uint64_t  x3, uint64_t  x4, uint64_t  x5,
    uint64_t  y0, uint64_t  y1, uint64_t  y2, uint64_t  y3, uint64_t  y4, uint64_t  y5
    )
{
    uint32_t
        xl0, xl1, xl2, xl3, xl4, xl5,
        xh0, xh1, xh2, xh3, xh4, xh5,
        yl0, yl1, yl2, yl3, yl4, yl5,
        yh0, yh1, yh2, yh3, yh4, yh5,
        l0, l1, h0, h1;

    uint64_t t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb;

    unpack(xl0, xh0, x0);
    unpack(xl1, xh1, x1);
    unpack(xl2, xh2, x2);
    unpack(xl3, xh3, x3);
    unpack(xl4, xh4, x4);
    unpack(xl5, xh5, x5);

    unpack(yl0, yh0, y0);
    unpack(yl1, yh1, y1);
    unpack(yl2, yh2, y2);
    unpack(yl3, yh3, y3);
    unpack(yl4, yh4, y4);
    unpack(yl5, yh5, y5);

    // xl*yl and xh*yh -> z

    mul_wide_u32(z0, xl0, yl0);
    mul_wide_u32(z1, xl0, yl1);
    mul_wide_u32(z2, xl0, yl2);
    mul_wide_u32(z3, xl0, yl3);
    mul_wide_u32(z4, xl0, yl4);
    mul_wide_u32(z5, xl0, yl5);

    // xl*yh and xh*yl -> t

    mul_wide_u32(t0, xl0, yh0);
    mul_wide_u32(t1, xl0, yh1);
    mul_wide_u32(t2, xl0, yh2);
    mul_wide_u32(t3, xl0, yh3);
    mul_wide_u32(t4, xl0, yh4);
    mul_wide_u32(t5, xl0, yh5);

    mad_wide_cc_u32 (z1, xh0, yh0, z1);
    madc_wide_cc_u32(z2, xh0, yh1, z2);
    madc_wide_cc_u32(z3, xh0, yh2, z3);
    madc_wide_cc_u32(z4, xh0, yh3, z4);
    madc_wide_cc_u32(z5, xh0, yh4, z5);
    madc_wide_u32   (z6, xh0, yh5,  0);

    mad_wide_cc_u32 (t0, xh0, yl0, t0);
    madc_wide_cc_u32(t1, xh0, yl1, t1);
    madc_wide_cc_u32(t2, xh0, yl2, t2);
    madc_wide_cc_u32(t3, xh0, yl3, t3);
    madc_wide_cc_u32(t4, xh0, yl4, t4);
    madc_wide_cc_u32(t5, xh0, yl5, t5);
    addc_u64        (t6, 0, 0);

    mad_wide_cc_u32 (z1, xl1, yl0, z1);
    madc_wide_cc_u32(z2, xl1, yl1, z2);
    madc_wide_cc_u32(z3, xl1, yl2, z3);
    madc_wide_cc_u32(z4, xl1, yl3, z4);
    madc_wide_cc_u32(z5, xl1, yl4, z5);
    madc_wide_cc_u32(z6, xl1, yl5, z6);
    addc_u64        (z7, 0, 0);

    mad_wide_cc_u32 (t1, xl1, yh0, t1);
    madc_wide_cc_u32(t2, xl1, yh1, t2);
    madc_wide_cc_u32(t3, xl1, yh2, t3);
    madc_wide_cc_u32(t4, xl1, yh3, t4);
    madc_wide_cc_u32(t5, xl1, yh4, t5);
    madc_wide_u32   (t6, xl1, yh5, t6);

    mad_wide_cc_u32 (z2, xh1, yh0, z2);
    madc_wide_cc_u32(z3, xh1, yh1, z3);
    madc_wide_cc_u32(z4, xh1, yh2, z4);
    madc_wide_cc_u32(z5, xh1, yh3, z5);
    madc_wide_cc_u32(z6, xh1, yh4, z6);
    madc_wide_cc_u32(z7, xh1, yh5, z7);

    mad_wide_cc_u32 (t1, xh1, yl0, t1);
    madc_wide_cc_u32(t2, xh1, yl1, t2);
    madc_wide_cc_u32(t3, xh1, yl2, t3);
    madc_wide_cc_u32(t4, xh1, yl3, t4);
    madc_wide_cc_u32(t5, xh1, yl4, t5);
    madc_wide_cc_u32(t6, xh1, yl5, t6);
    addc_u64        (t7, 0, 0);

    mad_wide_cc_u32 (z2, xl2, yl0, z2);
    madc_wide_cc_u32(z3, xl2, yl1, z3);
    madc_wide_cc_u32(z4, xl2, yl2, z4);
    madc_wide_cc_u32(z5, xl2, yl3, z5);
    madc_wide_cc_u32(z6, xl2, yl4, z6);
    madc_wide_cc_u32(z7, xl2, yl5, z7);
    addc_u64        (z8, 0, 0);

    mad_wide_cc_u32 (t2, xl2, yh0, t2);
    madc_wide_cc_u32(t3, xl2, yh1, t3);
    madc_wide_cc_u32(t4, xl2, yh2, t4);
    madc_wide_cc_u32(t5, xl2, yh3, t5);
    madc_wide_cc_u32(t6, xl2, yh4, t6);
    madc_wide_u32   (t7, xl2, yh5, t7);

    mad_wide_cc_u32 (z3, xh2, yh0, z3);
    madc_wide_cc_u32(z4, xh2, yh1, z4);
    madc_wide_cc_u32(z5, xh2, yh2, z5);
    madc_wide_cc_u32(z6, xh2, yh3, z6);
    madc_wide_cc_u32(z7, xh2, yh4, z7);
    madc_wide_cc_u32(z8, xh2, yh5, z8);

    mad_wide_cc_u32 (t2, xh2, yl0, t2);
    madc_wide_cc_u32(t3, xh2, yl1, t3);
    madc_wide_cc_u32(t4, xh2, yl2, t4);
    madc_wide_cc_u32(t5, xh2, yl3, t5);
    madc_wide_cc_u32(t6, xh2, yl4, t6);
    madc_wide_cc_u32(t7, xh2, yl5, t7);
    addc_u64        (t8, 0, 0);

    mad_wide_cc_u32 (z3, xl3, yl0, z3);
    madc_wide_cc_u32(z4, xl3, yl1, z4);
    madc_wide_cc_u32(z5, xl3, yl2, z5);
    madc_wide_cc_u32(z6, xl3, yl3, z6);
    madc_wide_cc_u32(z7, xl3, yl4, z7);
    madc_wide_cc_u32(z8, xl3, yl5, z8);
    addc_u64        (z9, 0, 0);

    mad_wide_cc_u32 (t3, xl3, yh0, t3);
    madc_wide_cc_u32(t4, xl3, yh1, t4);
    madc_wide_cc_u32(t5, xl3, yh2, t5);
    madc_wide_cc_u32(t6, xl3, yh3, t6);
    madc_wide_cc_u32(t7, xl3, yh4, t7);
    madc_wide_u32   (t8, xl3, yh5, t8);

    mad_wide_cc_u32 (z4, xh3, yh0, z4);
    madc_wide_cc_u32(z5, xh3, yh1, z5);
    madc_wide_cc_u32(z6, xh3, yh2, z6);
    madc_wide_cc_u32(z7, xh3, yh3, z7);
    madc_wide_cc_u32(z8, xh3, yh4, z8);
    madc_wide_cc_u32(z9, xh3, yh5, z9);

    mad_wide_cc_u32 (t3, xh3, yl0, t3);
    madc_wide_cc_u32(t4, xh3, yl1, t4);
    madc_wide_cc_u32(t5, xh3, yl2, t5);
    madc_wide_cc_u32(t6, xh3, yl3, t6);
    madc_wide_cc_u32(t7, xh3, yl4, t7);
    madc_wide_cc_u32(t8, xh3, yl5, t8);
    addc_u64        (t9, 0, 0);

    mad_wide_cc_u32 (z4, xl4, yl0, z4);
    madc_wide_cc_u32(z5, xl4, yl1, z5);
    madc_wide_cc_u32(z6, xl4, yl2, z6);
    madc_wide_cc_u32(z7, xl4, yl3, z7);
    madc_wide_cc_u32(z8, xl4, yl4, z8);
    madc_wide_cc_u32(z9, xl4, yl5, z9);
    addc_u64        (za, 0, 0);

    mad_wide_cc_u32 (t4, xl4, yh0, t4);
    madc_wide_cc_u32(t5, xl4, yh1, t5);
    madc_wide_cc_u32(t6, xl4, yh2, t6);
    madc_wide_cc_u32(t7, xl4, yh3, t7);
    madc_wide_cc_u32(t8, xl4, yh4, t8);
    madc_wide_u32   (t9, xl4, yh5, t9);

    mad_wide_cc_u32 (z5, xh4, yh0, z5);
    madc_wide_cc_u32(z6, xh4, yh1, z6);
    madc_wide_cc_u32(z7, xh4, yh2, z7);
    madc_wide_cc_u32(z8, xh4, yh3, z8);
    madc_wide_cc_u32(z9, xh4, yh4, z9);
    madc_wide_cc_u32(za, xh4, yh5, za);

    mad_wide_cc_u32 (t4, xh4, yl0, t4);
    madc_wide_cc_u32(t5, xh4, yl1, t5);
    madc_wide_cc_u32(t6, xh4, yl2, t6);
    madc_wide_cc_u32(t7, xh4, yl3, t7);
    madc_wide_cc_u32(t8, xh4, yl4, t8);
    madc_wide_cc_u32(t9, xh4, yl5, t9);
    addc_u64        (ta, 0, 0);

    mad_wide_cc_u32 (z5, xl5, yl0, z5);
    madc_wide_cc_u32(z6, xl5, yl1, z6);
    madc_wide_cc_u32(z7, xl5, yl2, z7);
    madc_wide_cc_u32(z8, xl5, yl3, z8);
    madc_wide_cc_u32(z9, xl5, yl4, z9);
    madc_wide_cc_u32(za, xl5, yl5, za);
    addc_u64        (zb, 0, 0);

    mad_wide_cc_u32 (t5, xl5, yh0, t5);
    madc_wide_cc_u32(t6, xl5, yh1, t6);
    madc_wide_cc_u32(t7, xl5, yh2, t7);
    madc_wide_cc_u32(t8, xl5, yh3, t8);
    madc_wide_cc_u32(t9, xl5, yh4, t9);
    madc_wide_u32   (ta, xl5, yh5, ta);

    mad_wide_cc_u32 (z6, xh5, yh0, z6);
    madc_wide_cc_u32(z7, xh5, yh1, z7);
    madc_wide_cc_u32(z8, xh5, yh2, z8);
    madc_wide_cc_u32(z9, xh5, yh3, z9);
    madc_wide_cc_u32(za, xh5, yh4, za);
    madc_wide_u32   (zb, xh5, yh5, zb);

    mad_wide_cc_u32 (t5, xh5, yl0, t5);
    madc_wide_cc_u32(t6, xh5, yl1, t6);
    madc_wide_cc_u32(t7, xh5, yl2, t7);
    madc_wide_cc_u32(t8, xh5, yl3, t8);
    madc_wide_cc_u32(t9, xh5, yl4, t9);
    madc_wide_cc_u32(ta, xh5, yl5, ta);
    addc_u64        (tb, 0, 0);

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
    unpack(l0, h0, zb);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, tb);  addc_cc_u32(h0, h0, l1); pack(zb, l0, h0);
}

#endif
// vim: ts=4 et sw=4 si
