// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "ptx.cuh"
#include "fp_mul.cuh"
#include "fp_reduce12.cuh"

/**
 * @brief Fp multiply-multiply-add. Fast execution of z = (v*w + x*y) mod p
 * The double-wide products are added before reduction, saving one reduction.
 *
 * @param[out] z
 * @param[in] v
 * @param[in] w
 * @param[in] x
 * @param[in] y
 * @return void
 */
__device__ void fp_mma(fp_t &z, const fp_t &v, const fp_t &w, const fp_t &x, const fp_t &y) {
#if 1
    fp_t t, u;
    fp_mul(t, v, w);
    fp_mul(u, x, y);
    fp_add(z, t, u);
#else
    uint64_t
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc,
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, ua, ub,
        v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3], v4 = v[4], v5 = v[5],
        w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3], w4 = w[4], w5 = w[5],
        x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5],
        y0 = y[0], y1 = y[1], y2 = y[2], y3 = y[3], y4 = y[4], y5 = y[5],
        z0, z1, z2, z3, z4, z5;

    fp_print("v ", v);
    fp_print("w ", w);
    fp_print("x ", x);
    fp_print("y ", y);

    // t = v * w
    fp_mul(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, v0, v1, v2, v3, v4, v5, w0, w1, w2, w3, w4, w5);

    printf("t #x%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx\n",
        tb, ta, t9, t8, t7, t6, t5, t4, t3, t2, t1, t0);

    // u = x * y
    fp_mul(u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, ua, ub, x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5);

    printf("u #x%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx%016lx\n",
        ub, ua, u9, u8, u7, u6, u5, u4, u3, u2, u1, u0);

    add_cc_u64 (t0, t0, u0);
    addc_cc_u64(t1, t1, u1);
    addc_cc_u64(t2, t2, u2);
    addc_cc_u64(t3, t3, u3);
    addc_cc_u64(t4, t4, u4);
    addc_cc_u64(t5, t5, u5);
    addc_cc_u64(t6, t6, u6);
    addc_cc_u64(t7, t7, u7);
    addc_cc_u64(t8, t8, u8);
    addc_cc_u64(t9, t9, u9);
    addc_cc_u64(ta, ta, ua);
    addc_cc_u64(tb, tb, ub);
    addc_u64   (tc,  0,  0);
#if 1
    if (tc > 0) {
        sub_cc_u64 (t6, t6, 0x89f6fffffffd0003U);
        subc_cc_u64(t7, t7, 0x140bfff43bf3fffdU);
        subc_cc_u64(t8, t8, 0xa0b767a8ac38a745U);
        subc_cc_u64(t9, t9, 0x8831a7ac8fada8baU);
        subc_cc_u64(ta, ta, 0xa3f8e5685da91392U);
        subc_cc_u64(tb, tb, 0xea09a13c057f1b6cU);
        subc_u64   (tc, tc, 0);

        if (tc > 0) {
            sub_cc_u64 (t6, t6, 0x89f6fffffffd0003U);
            subc_cc_u64(t7, t7, 0x140bfff43bf3fffdU);
            subc_cc_u64(t8, t8, 0xa0b767a8ac38a745U);
            subc_cc_u64(t9, t9, 0x8831a7ac8fada8baU);
            subc_cc_u64(ta, ta, 0xa3f8e5685da91392U);
            subc_cc_u64(tb, tb, 0xea09a13c057f1b6cU);
            subc_u64   (tc, tc, 0);
        }
    }
#else
    uint64_t uc;

    sub_cc_u64 (u6, t6, 0x89f6fffffffd0003U);
    subc_cc_u64(u7, t7, 0x140bfff43bf3fffdU);
    subc_cc_u64(u8, t8, 0xa0b767a8ac38a745U);
    subc_cc_u64(u9, t9, 0x8831a7ac8fada8baU);
    subc_cc_u64(ua, ta, 0xa3f8e5685da91392U);
    subc_cc_u64(ub, tb, 0xea09a13c057f1b6cU);
    subc_u64   (uc, tc, 0);

    t6 = tc > 0 ? u6 : t6;
    t7 = tc > 0 ? u7 : t7;
    t8 = tc > 0 ? u8 : t8;
    t9 = tc > 0 ? u9 : t9;
    ta = tc > 0 ? ua : ta;
    tb = tc > 0 ? ub : tb;
    tc = tc > 0 ? uc : tc;

    sub_cc_u64 (u6, t6, 0x89f6fffffffd0003U);
    subc_cc_u64(u7, t7, 0x140bfff43bf3fffdU);
    subc_cc_u64(u8, t8, 0xa0b767a8ac38a745U);
    subc_cc_u64(u9, t9, 0x8831a7ac8fada8baU);
    subc_cc_u64(ua, ta, 0xa3f8e5685da91392U);
    subc_cc_u64(ub, tb, 0xea09a13c057f1b6cU);
    subc_u64   (uc, tc, 0);

    t6 = tc > 0 ? u6 : t6;
    t7 = tc > 0 ? u7 : t7;
    t8 = tc > 0 ? u8 : t8;
    t9 = tc > 0 ? u9 : t9;
    ta = tc > 0 ? ua : ta;
    tb = tc > 0 ? ub : tb;
    tc = tc > 0 ? uc : tc;

#endif

    fp_reduce12(z0, z1, z2, z3, z4, z5, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb);
    fp_print("z ", z);
#endif
}

// vim: ts=4 et sw=4 si
