// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_REDUCE12_CUH
#define FP_REDUCE12_CUH

#include <cstdio>

#include "fp.cuh"
#include "ptx.cuh"

#define mB 0x1A0111EA
#define mA 0x397FE69A
#define m9 0x4B1BA7B6
#define m8 0x434BACD7
#define m7 0x64774B84
#define m6 0xF38512BF
#define m5 0x6730D2A0
#define m4 0xF6B0F624
#define m3 0x1EABFFFE
#define m2 0xB153FFFF
#define m1 0xB9FEFFFF
#define m0 0xFFFFAAAB

#define muC 0x00000009U
#define muB 0xD835D2F3U
#define muA 0xCC9E45CEU
#define mu9 0x28101B0CU
#define mu8 0xC7A6BA29U
#define mu7 0x1B82741FU
#define mu6 0xF6A0A94BU
#define mu5 0xDF4771E0U
#define mu4 0x286779D3U
#define mu3 0x997167A0U
#define mu2 0x58F1C07BU
#define mu1 0x13E207F5U
#define mu0 0x6591BA2EU

__device__ __forceinline__ void fp_reduce12(
    uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3, uint64_t &z4, uint64_t &z5,
    uint64_t  x0, uint64_t  x1, uint64_t  x2, uint64_t  x3, uint64_t  x4, uint64_t  x5,
    uint64_t  x6, uint64_t  x7, uint64_t  x8, uint64_t  x9, uint64_t  xa, uint64_t  xb
    )
{
    uint32_t
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb,
        tc, td, te, tf, tg, th, ti, tj, tk, tl, tm, tn,
        l0, l1, h0, h1;

    uint64_t
        q0, q1, q2, q3, q4, q5, q6,
        r0, r1, r2, r3, r4, r5, r6,
        u0, u1, u2, u3, u4, u5, u6;

    unpack(t0, t1, x0);
    unpack(t2, t3, x1);
    unpack(t4, t5, x2);
    unpack(t6, t7, x3);
    unpack(t8, t9, x4);
    unpack(ta, tb, x5);
    unpack(tc, td, x6);
    unpack(te, tf, x7);
    unpack(tg, th, x8);
    unpack(ti, tj, x9);
    unpack(tk, tl, xa);
    unpack(tm, tn, xb);

    mul_wide_u32(u0, muC, tb);
    mul_wide_u32(q0, muC, tc);


    mad_wide_cc_u32 (u0, muB, tc, u0);
    addc_u64        (u1, 0, 0);

    mad_wide_cc_u32 (q0, muB, td, q0);
    addc_u64        (q1, 0, 0);


    mad_wide_cc_u32(u0, muA, td, u0);
    madc_wide_u32  (u1, muC, td, u1);

    mad_wide_cc_u32(q0, muA, te, q0);
    madc_wide_u32  (q1, muC, te, q1);


    mad_wide_cc_u32 (u0, mu9, te, u0);
    madc_wide_cc_u32(u1, muB, te, u1);
    addc_u64        (u2, 0, 0);

    mad_wide_cc_u32 (q0, mu9, tf, q0);
    madc_wide_cc_u32(q1, muB, tf, q1);
    addc_u64        (q2, 0, 0);


    mad_wide_cc_u32 (u0, mu8, tf, u0);
    madc_wide_cc_u32(u1, muA, tf, u1);
    madc_wide_u32   (u2, muC, tf, u2);

    mad_wide_cc_u32 (q0, mu8, tg, q0);
    madc_wide_cc_u32(q1, muA, tg, q1);
    madc_wide_u32   (q2, muC, tg, q2);


    mad_wide_cc_u32 (u0, mu7, tg, u0);
    madc_wide_cc_u32(u1, mu9, tg, u1);
    madc_wide_cc_u32(u2, muB, tg, u2);
    addc_u64        (u3, 0, 0);

    mad_wide_cc_u32 (q0, mu7, th, q0);
    madc_wide_cc_u32(q1, mu9, th, q1);
    madc_wide_cc_u32(q2, muB, th, q2);
    addc_u64        (q3, 0, 0);


    mad_wide_cc_u32 (u0, mu6, th, u0);
    madc_wide_cc_u32(u1, mu8, th, u1);
    madc_wide_cc_u32(u2, muA, th, u2);
    madc_wide_u32   (u3, muC, th, u3);

    mad_wide_cc_u32 (q0, mu6, ti, q0);
    madc_wide_cc_u32(q1, mu8, ti, q1);
    madc_wide_cc_u32(q2, muA, ti, q2);
    madc_wide_u32   (q3, muC, ti, q3);


    mad_wide_cc_u32 (u0, mu5, ti, u0);
    madc_wide_cc_u32(u1, mu7, ti, u1);
    madc_wide_cc_u32(u2, mu9, ti, u2);
    madc_wide_cc_u32(u3, muB, ti, u3);
    addc_u64        (u4, 0, 0);

    mad_wide_cc_u32 (q0, mu5, tj, q0);
    madc_wide_cc_u32(q1, mu7, tj, q1);
    madc_wide_cc_u32(q2, mu9, tj, q2);
    madc_wide_cc_u32(q3, muB, tj, q3);
    addc_u64        (q4, 0, 0);


    mad_wide_cc_u32 (u0, mu4, tj, u0);
    madc_wide_cc_u32(u1, mu6, tj, u1);
    madc_wide_cc_u32(u2, mu8, tj, u2);
    madc_wide_cc_u32(u3, muA, tj, u3);
    madc_wide_u32   (u4, muC, tj, u4);

    mad_wide_cc_u32 (q0, mu4, tk, q0);
    madc_wide_cc_u32(q1, mu6, tk, q1);
    madc_wide_cc_u32(q2, mu8, tk, q2);
    madc_wide_cc_u32(q3, muA, tk, q3);
    madc_wide_u32   (q4, muC, tk, q4);


    mad_wide_cc_u32 (u0, mu3, tk, u0);
    madc_wide_cc_u32(u1, mu5, tk, u1);
    madc_wide_cc_u32(u2, mu7, tk, u2);
    madc_wide_cc_u32(u3, mu9, tk, u3);
    madc_wide_cc_u32(u4, muB, tk, u4);
    addc_u64        (u5, 0, 0);

    mad_wide_cc_u32 (q0, mu3, tl, q0);
    madc_wide_cc_u32(q1, mu5, tl, q1);
    madc_wide_cc_u32(q2, mu7, tl, q2);
    madc_wide_cc_u32(q3, mu9, tl, q3);
    madc_wide_cc_u32(q4, muB, tl, q4);
    addc_u64        (q5, 0, 0);


    mad_wide_cc_u32 (u0, mu2, tl, u0);
    madc_wide_cc_u32(u1, mu4, tl, u1);
    madc_wide_cc_u32(u2, mu6, tl, u2);
    madc_wide_cc_u32(u3, mu8, tl, u3);
    madc_wide_cc_u32(u4, muA, tl, u4);
    madc_wide_u32   (u5, muC, tl, u5);

    mad_wide_cc_u32 (q0, mu2, tm, q0);
    madc_wide_cc_u32(q1, mu4, tm, q1);
    madc_wide_cc_u32(q2, mu6, tm, q2);
    madc_wide_cc_u32(q3, mu8, tm, q3);
    madc_wide_cc_u32(q4, muA, tm, q4);
    madc_wide_u32   (q5, muC, tm, q5);


    mad_wide_cc_u32 (u0, mu1, tm, u0);
    madc_wide_cc_u32(u1, mu3, tm, u1);
    madc_wide_cc_u32(u2, mu5, tm, u2);
    madc_wide_cc_u32(u3, mu7, tm, u3);
    madc_wide_cc_u32(u4, mu9, tm, u4);
    madc_wide_cc_u32(u5, muB, tm, u5);
    addc_u64        (u6, 0, 0);

    mad_wide_cc_u32 (q0, mu1, tn, q0);
    madc_wide_cc_u32(q1, mu3, tn, q1);
    madc_wide_cc_u32(q2, mu5, tn, q2);
    madc_wide_cc_u32(q3, mu7, tn, q3);
    madc_wide_cc_u32(q4, mu9, tn, q4);
    madc_wide_cc_u32(q5, muB, tn, q5);
    addc_u64        (q6, 0, 0);


    mad_wide_cc_u32 (u0, mu0, tn, u0);
    madc_wide_cc_u32(u1, mu2, tn, u1);
    madc_wide_cc_u32(u2, mu4, tn, u2);
    madc_wide_cc_u32(u3, mu6, tn, u3);
    madc_wide_cc_u32(u4, mu8, tn, u4);
    madc_wide_cc_u32(u5, muA, tn, u5);
    madc_wide_u32   (u6, muC, tn, u6);

    //////////////////////////////////
    // q += u >> 32
    //////////////////////////////////

    unpack(l0, h0, u0);                           unpack(l1, h1, q0);  add_cc_u32 (h0, h0, l1); pack(q0, l0, h0);
    unpack(l0, h0, u1);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q1);  addc_cc_u32(h0, h0, l1); pack(q1, l0, h0);
    unpack(l0, h0, u2);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q2);  addc_cc_u32(h0, h0, l1); pack(q2, l0, h0);
    unpack(l0, h0, u3);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q3);  addc_cc_u32(h0, h0, l1); pack(q3, l0, h0);
    unpack(l0, h0, u4);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q4);  addc_cc_u32(h0, h0, l1); pack(q4, l0, h0);
    unpack(l0, h0, u5);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q5);  addc_cc_u32(h0, h0, l1); pack(q5, l0, h0);
    unpack(l0, h0, u6);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, q6);  addc_cc_u32(h0, h0, l1); pack(q6, l0, h0);

    //////////////////////////////////
    // r = q * m mod 2^416
    //////////////////////////////////

    unpack(l0, h0, q0);

    mul_wide_u32(r0, m0, h0);
    mul_wide_u32(r1, m2, h0);
    mul_wide_u32(r2, m4, h0);
    mul_wide_u32(r3, m6, h0);
    mul_wide_u32(r4, m8, h0);
    mul_wide_u32(r5, mA, h0);

    mul_wide_u32(u0, m1, h0);
    mul_wide_u32(u1, m3, h0);
    mul_wide_u32(u2, m5, h0);
    mul_wide_u32(u3, m7, h0);
    mul_wide_u32(u4, m9, h0);
    mul_wide_u32(u5, mB, h0);

    unpack(l0, h0, q1);

    mad_wide_cc_u32 (r1, m1, l0, r1);
    madc_wide_cc_u32(r2, m3, l0, r2);
    madc_wide_cc_u32(r3, m5, l0, r3);
    madc_wide_cc_u32(r4, m7, l0, r4);
    madc_wide_cc_u32(r5, m9, l0, r5);
    madc_wide_u32   (r6, mB, l0,  0);

    mad_wide_cc_u32 (u0, m0, l0, u0);
    madc_wide_cc_u32(u1, m2, l0, u1);
    madc_wide_cc_u32(u2, m4, l0, u2);
    madc_wide_cc_u32(u3, m6, l0, u3);
    madc_wide_cc_u32(u4, m8, l0, u4);
    madc_wide_u32   (u5, mA, l0, u5);

    mad_wide_cc_u32 (r1, m0, h0, r1);
    madc_wide_cc_u32(r2, m2, h0, r2);
    madc_wide_cc_u32(r3, m4, h0, r3);
    madc_wide_cc_u32(r4, m6, h0, r4);
    madc_wide_cc_u32(r5, m8, h0, r5);
    madc_wide_u32   (r6, mA, h0, r6);

    mad_wide_cc_u32 (u1, m1, h0, u1);
    madc_wide_cc_u32(u2, m3, h0, u2);
    madc_wide_cc_u32(u3, m5, h0, u3);
    madc_wide_cc_u32(u4, m7, h0, u4);
    madc_wide_u32   (u5, m9, h0, u5);

    unpack(l0, h0, q2);

    mad_wide_cc_u32 (r2, m1, l0, r2);
    madc_wide_cc_u32(r3, m3, l0, r3);
    madc_wide_cc_u32(r4, m5, l0, r4);
    madc_wide_cc_u32(r5, m7, l0, r5);
    madc_wide_u32   (r6, m9, l0, r6);

    mad_wide_cc_u32 (u1, m0, l0, u1);
    madc_wide_cc_u32(u2, m2, l0, u2);
    madc_wide_cc_u32(u3, m4, l0, u3);
    madc_wide_cc_u32(u4, m6, l0, u4);
    madc_wide_u32   (u5, m8, l0, u5);

    mad_wide_cc_u32 (r2, m0, h0, r2);
    madc_wide_cc_u32(r3, m2, h0, r3);
    madc_wide_cc_u32(r4, m4, h0, r4);
    madc_wide_cc_u32(r5, m6, h0, r5);
    madc_wide_u32   (r6, m8, h0, r6);

    mad_wide_cc_u32 (u2, m1, h0, u2);
    madc_wide_cc_u32(u3, m3, h0, u3);
    madc_wide_cc_u32(u4, m5, h0, u4);
    madc_wide_u32   (u5, m7, h0, u5);

    unpack(l0, h0, q3);

    mad_wide_cc_u32 (r3, m1, l0, r3);
    madc_wide_cc_u32(r4, m3, l0, r4);
    madc_wide_cc_u32(r5, m5, l0, r5);
    madc_wide_u32   (r6, m7, l0, r6);

    mad_wide_cc_u32 (u2, m0, l0, u2);
    madc_wide_cc_u32(u3, m2, l0, u3);
    madc_wide_cc_u32(u4, m4, l0, u4);
    madc_wide_u32   (u5, m6, l0, u5);

    mad_wide_cc_u32 (r3, m0, h0, r3);
    madc_wide_cc_u32(r4, m2, h0, r4);
    madc_wide_cc_u32(r5, m4, h0, r5);
    madc_wide_u32   (r6, m6, h0, r6);

    mad_wide_cc_u32 (u3, m1, h0, u3);
    madc_wide_cc_u32(u4, m3, h0, u4);
    madc_wide_u32   (u5, m5, h0, u5);

    unpack(l0, h0, q4);

    mad_wide_cc_u32 (r4, m1, l0, r4);
    madc_wide_cc_u32(r5, m3, l0, r5);
    madc_wide_u32   (r6, m5, l0, r6);

    mad_wide_cc_u32 (u3, m0, l0, u3);
    madc_wide_cc_u32(u4, m2, l0, u4);
    madc_wide_u32   (u5, m4, l0, u5);

    mad_wide_cc_u32 (r4, m0, h0, r4);
    madc_wide_cc_u32(r5, m2, h0, r5);
    madc_wide_u32   (r6, m4, h0, r6);

    mad_wide_cc_u32 (u4, m1, h0, u4);
    madc_wide_u32   (u5, m3, h0, u5);

    unpack(l0, h0, q5);

    mad_wide_cc_u32 (r5, m1, l0, r5);
    madc_wide_u32   (r6, m3, l0, r6);

    mad_wide_cc_u32 (u4, m0, l0, u4);
    madc_wide_u32   (u5, m2, l0, u5);

    mad_wide_cc_u32 (r5, m0, h0, r5);
    madc_wide_u32   (r6, m2, h0, r6);

    mad_wide_u32    (u5, m1, h0, u5);

    unpack(l0, h0, q6);

    mad_wide_cc_u32 (r6, m1, l0, r6);

    mad_wide_cc_u32 (u5, m0, l0, u5);

    mad_wide_cc_u32 (r6, m0, h0, r6);

    //////////////////////////////////
    // r += u << 32
    // r %= 1 << 416
    //////////////////////////////////

    unpack(l0, h0, r0);                           unpack(l1, h1, u0);  add_cc_u32 (h0, h0, l1); pack(r0, l0, h0);
    unpack(l0, h0, r1);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u1);  addc_cc_u32(h0, h0, l1); pack(r1, l0, h0);
    unpack(l0, h0, r2);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u2);  addc_cc_u32(h0, h0, l1); pack(r2, l0, h0);
    unpack(l0, h0, r3);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u3);  addc_cc_u32(h0, h0, l1); pack(r3, l0, h0);
    unpack(l0, h0, r4);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u4);  addc_cc_u32(h0, h0, l1); pack(r4, l0, h0);
    unpack(l0, h0, r5);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u5);  addc_cc_u32(h0, h0, l1); pack(r5, l0, h0);
    unpack(l0, h0, r6);  addc_cc_u32(l0, l0, h1); unpack(l1, h1, u6);  addc_cc_u32(h0, h0, l1); pack(r6, l0,  0);

    //////////////////////////////////
    // r = (x % (1 << 416)) - r
    //////////////////////////////////

    unpack(l0, h0, r0); sub_cc_u32 (t0, t0, l0); subc_cc_u32(t1, t1, h0); pack(z0, t0, t1);
    unpack(l0, h0, r1); subc_cc_u32(t2, t2, l0); subc_cc_u32(t3, t3, h0); pack(z1, t2, t3);
    unpack(l0, h0, r2); subc_cc_u32(t4, t4, l0); subc_cc_u32(t5, t5, h0); pack(z2, t4, t5);
    unpack(l0, h0, r3); subc_cc_u32(t6, t6, l0); subc_cc_u32(t7, t7, h0); pack(z3, t6, t7);
    unpack(l0, h0, r4); subc_cc_u32(t8, t8, l0); subc_cc_u32(t9, t9, h0); pack(z4, t8, t9);
    unpack(l0, h0, r5); subc_cc_u32(ta, ta, l0); subc_cc_u32(tb, tb, h0); pack(z5, ta, tb);
    unpack(l0, h0, r6); subc_cc_u32(tc, tc, l0);                          pack(r6, tc,  0);
}

#endif
// vim: ts=4 et sw=4 si
