// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_REDUCE7_CUH
#define FP_REDUCE7_CUH

#include <cassert>
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

#define mu 0x9d

__device__ __forceinline__ void fp_reduce7(
    uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3, uint64_t &z4, uint64_t &z5,
    uint64_t  x0, uint64_t  x1, uint64_t  x2, uint64_t  x3, uint64_t  x4, uint64_t  x5, uint64_t x6
    )
{
    assert(x6 < 12);    // This function is only intended for multiplying a 6-word residue by a constant <= 12

    uint32_t
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td,
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, la, lb,
        h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, ha, hb;

    uint64_t
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, ua, ub;

    unpack(t0, t1, x0);
    unpack(t2, t3, x1);
    unpack(t4, t5, x2);
    unpack(t6, t7, x3);
    unpack(t8, t9, x4);
    unpack(ta, tb, x5);
    unpack(tc, td, x6); // the value in td is not used 

    shf_l_wrap_b32(td, tb, tc, 4);  // q1

    mul_lo_u32(td, td, mu);         // q2

    shr_b32(td, td, 8);             // q3

    assert(td < 0x80);

    // q3 * m

    mul_wide_u32(u0, td, m0);
    mul_wide_u32(u1, td, m1);
    mul_wide_u32(u2, td, m2);
    mul_wide_u32(u3, td, m3);
    mul_wide_u32(u4, td, m4);
    mul_wide_u32(u5, td, m5);
    mul_wide_u32(u6, td, m6);
    mul_wide_u32(u7, td, m7);
    mul_wide_u32(u8, td, m8);
    mul_wide_u32(u9, td, m9);
    mul_wide_u32(ua, td, mA);
    mul_wide_u32(ub, td, mB);

    unpack(l0, h0, u0);
    unpack(l1, h1, u1);
    unpack(l2, h2, u2);
    unpack(l3, h3, u3);
    unpack(l4, h4, u4);
    unpack(l5, h5, u5);
    unpack(l6, h6, u6);
    unpack(l7, h7, u7);
    unpack(l8, h8, u8);
    unpack(l9, h9, u9);
    unpack(la, ha, ua);
    unpack(lb, hb, ub);

    add_cc_u32 (h0, h0, l1);
    addc_cc_u32(l2, l2, h1);
    addc_cc_u32(h2, h2, l3);
    addc_cc_u32(l4, l4, h3);
    addc_cc_u32(h4, h4, l5);
    addc_cc_u32(l6, l6, h5);
    addc_cc_u32(h6, h6, l7);
    addc_cc_u32(l8, l8, h7);
    addc_cc_u32(h8, h8, l9);
    addc_cc_u32(la, la, h9);
    addc_u32   (ha, ha, lb);

    pack(u0, l0, h0);
    pack(u1, l2, h2);
    pack(u2, l4, h4);
    pack(u3, l6, h6);
    pack(u4, l8, h8);
    pack(u5, la, ha);

    // r = r1 - r2 = x - q3 mod 2^384

    sub_cc_u64 (z0, x0, u0);
    subc_cc_u64(z1, x1, u1);
    subc_cc_u64(z2, x2, u2);
    subc_cc_u64(z3, x3, u3);
    subc_cc_u64(z4, x4, u4);
    subc_u64   (z5, x5, u5);
}

#undef mB
#undef mA
#undef m9
#undef m8
#undef m7
#undef m6
#undef m5
#undef m4
#undef m3
#undef m2
#undef m1
#undef m0

#undef mu

#endif
// vim: ts=4 et sw=4 si
