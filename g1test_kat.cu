// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "g1.cuh"
#include "g1test.cuh"

__managed__ g1p_t
    g1p_x0 = {
        { 0, 0, 0, 0, 0, 0 },
#if G1P_ANYINF
        { 1, 2, 3, 4, 5, 6 },
#else
        { 1, 0, 0, 0, 0, 0 },
#endif
        { 0, 0, 0, 0, 0, 0 },
    },
    g1p_x2 = {
        { 0x8a13c2b29f4325ad, 0x7803e16723f9f147, 0x6e7f23d4350c60bd, 0x062e48a6104cc52f, 0x9b6d4dac3f33e92c, 0x05dff4ac6726c6cb, }, // 0x05dff4ac6726c6cb9b6d4dac3f33e92c062e48a6104cc52f6e7f23d4350c60bd7803e16723f9f1478a13c2b29f4325ad
        { 0x2aa4d8e1b1df7ca5, 0x599a8782a1bea48a, 0x395952af0b6a0dbd, 0xd6093a00bb3e2bc9, 0x3c604c0410e5fc01, 0x14e4b429606d02bc, }, // 0x14e4b429606d02bc3c604c0410e5fc01d6093a00bb3e2bc9395952af0b6a0dbd599a8782a1bea48a2aa4d8e1b1df7ca5
        { 0x770553ef06aadf22, 0xc04c720ae01952ac, 0x9a55bee62edcdb27, 0x962f5650798fdf27, 0x28180e61b1f2cb8f, 0x0430df56ea4aba69, }, // 0x0430df56ea4aba6928180e61b1f2cb8f962f5650798fdf279a55bee62edcdb27c04c720ae01952ac770553ef06aadf22
    },
    g1p_x3 = {
        { 0x7f0318e712da7559, 0x266ac7358912af30, 0xc4374a5888cfca69, 0x4f376827946368db, 0x61156a0c4d426519, 0x0bba0304fb3212a4, }, // 0x0bba0304fb3212a461156a0c4d4265194f376827946368dbc4374a5888cfca69266ac7358912af307f0318e712da7559
        { 0xc25f47067af44e76, 0xcde09eab0276a0f4, 0xe9ec335b039fcf17, 0x727ce462858d3730, 0xdf4d86ed009f83fe, 0x0745d51f4d0912b3, }, // 0x0745d51f4d0912b3df4d86ed009f83fe727ce462858d3730e9ec335b039fcf17cde09eab0276a0f4c25f47067af44e76
        { 0xceb902463027454d, 0xc9d68f804d8ec369, 0xc2ddd8e251d7339c, 0xd787b07101270da7, 0xfc3d86788b163753, 0x191fe9e914d73631, }, // 0x191fe9e914d73631fc3d86788b163753d787b07101270da7c2ddd8e251d7339cc9d68f804d8ec369ceb902463027454d
    },
    g1p_x24 = {
        { 0xc8ff9a5471e72c92, 0x99683253e5aefa15, 0x5d6b135ab656eb43, 0x1b3776dc534fa4ab, 0xc2bfc4ab80c05017, 0x17b787b9910f9fa6, }, // 0x17b787b9910f9fa6c2bfc4ab80c050171b3776dc534fa4ab5d6b135ab656eb4399683253e5aefa15c8ff9a5471e72c92
        { 0xf275ddf2d8723a25, 0xce36e492230ed9cd, 0xae724c9b9d46d006, 0x5d4cec21d5949cc3, 0x9ce9b30542ce5589, 0x05f77ff79a5b6f8a, }, // 0x05f77ff79a5b6f8a9ce9b30542ce55895d4cec21d5949cc3ae724c9b9d46d006ce36e492230ed9cdf275ddf2d8723a25
        { 0x1eab6a2bf6bfeb17, 0xd88225cb44eaa0fb, 0x659281132d662bf8, 0xf0ac9c552dfd6f39, 0xb14437f70cc0f519, 0x0ae6513795046382, }, // 0x0ae6513795046382b14437f70cc0f519f0ac9c552dfd6f39659281132d662bf8d88225cb44eaa0fb1eab6a2bf6bfeb17
    },
    g1p_x25 = {
        { 0x2e4f86255524abb3, 0xebb6095fb99f8e97, 0x3a6ab2001ab4f83c, 0x606df6ee661d3aa2, 0xf6b369b6a22b4047, 0x0b1416f427fc4c5f, }, // 0x0b1416f427fc4c5ff6b369b6a22b4047606df6ee661d3aa23a6ab2001ab4f83cebb6095fb99f8e972e4f86255524abb3
        { 0xd0ee94cbc992ab24, 0x9fb5593bc61cd5bd, 0xc338a8acaef74389, 0x3a7da17eb290de91, 0xac616f60ea15f632, 0x0fc28d919f8ada25, }, // 0x0fc28d919f8ada25ac616f60ea15f6323a7da17eb290de91c338a8acaef743899fb5593bc61cd5bdd0ee94cbc992ab24
        { 0x1bb5a6833e3677ae, 0xf3d50cd096cd2ceb, 0xa1d2c3cbc5527a6e, 0x60613c9426b3b9a1, 0xee0f3f71173f041c, 0x139ca4dd9f299816, }, // 0x139ca4dd9f299816ee0f3f71173f041c60613c9426b3b9a1a1d2c3cbc5527a6ef3d50cd096cd2ceb1bb5a6833e3677ae
    };

/**
 * @brief Test operation over G1 using KAT and self consistency:
 * 
 * inf==inf
 * inf+inf == inf
 * G+0 == 0+G == G
 * G+G == 2*G
 * 2*G == 2*G with KAT
 * G+2*G == 3*G with KAT
 * 2*2*2*3G == 24G with KAT
 * 24G-2G+3G == 25G with KAT
 * 25*G == 25G with KAT
 * addsub(2G, G) == 3G, G with KAT
 * addsub(G, G) = (2G, 2G) (dbl and add)
 * 
 * @return void 
 */
__global__ void G1TestKAT(testval_t *) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    g1p_t p, q, r, t, u, v;

    // 0=0

    if (pass) {

        g1p_inf(p);     // p  = 0

        if (g1p_neq(p, g1p_x0)) {
            pass = false;

            g1p_print("0=0: FAIL: 0 !=  ", p);
        }
        ++count;
    }

    // 0+0

    if (pass) {

        g1p_inf(p);     // p  = 0
        g1p_inf(q);     // q  = 0
        g1p_add(p, g1p_x0);  // p += 0

        if (g1p_neq(p, q)) {
            pass = false;

            g1p_print("0+0: FAIL: 0 !=  ", p);
        }
        ++count;
    }

    // G+0 = 0+G = G

    if (pass) {

        g1p_inf(p);         // p  = 0
        g1p_gen(q);         // q  = G
        g1p_gen(r);         // r  = G

        g1p_add(p, q);      // p += G
        g1p_add(q, g1p_x0); // q += 0

        if (g1p_neq(p, r) || g1p_neq(q, r)) {
            pass = false;

            g1p_print("0+G: =  ", p);
            g1p_print("G+0: =  ", q);
            g1p_print("  G: =  ", r);
        }
        ++count;
    }

    // G+G

    if (pass) {

        g1p_gen(p);     // p  = G
        g1p_gen(q);     // q  = G
        g1p_add(p, q);  // p += q

        if (g1p_neq(p, g1p_x2)) {
            pass = false;

            g1p_print("G+G: FAIL: 2G !=  ", p);
        }
        ++count;
    }

    // 2G

    if (pass) {

        g1p_gen(p); // p  = G
        g1p_dbl(p); // p *= 2

        if (g1p_neq(p, g1p_x2)) {
            pass = false;

            g1p_print("2G FAIL: 2G !=  ", p);
        }
        ++count;
    }

    // G+2G

    if (pass) {

        g1p_gen(p);         // p  = G
        g1p_add(p, g1p_x2); // p += 2G

        if (g1p_neq(p, g1p_x3)) {
            pass = false;

            g1p_print("G+2G FAIL: 3G !=  ", p);
        }
        ++count;
    }

    // 2*2*2*3G = 24G

    if (pass) {

        g1p_dbl(p); // 2G
        g1p_dbl(p); // 2G
        g1p_dbl(p); // 2G

        if (g1p_neq(p, g1p_x24)) {
            pass = false;

            g1p_print("2*2*2*3G FAIL: 24G !=  ", p);
        }
        ++count;
    }

    // 24G-2G+3G = 25G

    if (pass) {

        g1p_sub(p, g1p_x2); // 22G
        g1p_add(p, g1p_x3); // 25G

        if (g1p_neq(p, g1p_x25)) {
            pass = false;

            g1p_print("24G-2G+3G FAIL: 25G !=  ", p);
        }
        ++count;
    }

    // 25 * G

    if (pass) {

        uint64_t x25[] = { 25, 0, 0, 0 };

        g1p_gen(p);
        g1p_mul(p, x25); // 25G

        if (g1p_neq(p, g1p_x25)) {
            pass = false;

            g1p_print("25G FAIL: 25G !=  ", p);
        }
        ++count;
    }

    // 2G+G, 2G-G

    if (pass) {

        g1p_cpy(p, g1p_x2); // 2G
        g1p_cpy(r, g1p_x2); // 2G

        g1p_gen(q); // 1G

        // Add-only reference
        g1p_add(p, q);

        // Sub-only reference
        g1p_sub(r, q);

        g1p_gen(p);
        g1p_cpy(p, g1p_x2); // 2G

        // Add & subtract
        g1p_addsub(p, q);

        if (g1p_neq(p, g1p_x3)) {
            pass = false;
            g1p_print("FAIL: 2G+G !=  ", p);
        }

        g1p_gen(p);

        if (g1p_neq(q, p)) {
            pass = false;

            g1p_print("FAIL: 2G-G !=  ", q);
        }
        ++count;
    }

    if (pass) {

        g1p_gen(p); // 1G
        g1p_gen(q); // 1G
        g1p_gen(r); // 1G

        for (int i=0; pass && i<20000; i++) {
            g1p_cpy(t, p);
            g1p_cpy(u, q);
            g1p_cpy(v, r);

            g1p_addsub(p, q);
            g1p_addsub(p, q);
            g1p_dbl(r);

            if (g1p_neq(p, q) || g1p_neq(q,r)) {
                pass = false;

                printf("FAIL after %d ok:\n", i);
                g1p_print("t =  ", t);
                g1p_print("u =  ", u);
                g1p_print("v =  ", v);
                g1p_print("p =  ", p);
                g1p_print("q =  ", q);
                g1p_print("r =  ", r);
            }
            ++count;
        }
    }

    if ((blockIdx.x | blockIdx.y | blockIdx.z | threadIdx.x | threadIdx.y | threadIdx.z) == 0)
    {
        printf("%ld test%s\n", count, count == 1 ? "" : "s");

        PRINTPASS(pass);
    }
}

// vim: ts=4 et sw=4 si
