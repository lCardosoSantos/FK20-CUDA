// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

__global__ void FpTestKAT(testval_t *) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    // fp_cpy

    if (pass) {
        __const__ uint64_t
            q[][6] = {
                { 0x94ec878d1a4356b3, 0xdc7f34e2dd5fc042, 0x9d419b78b22bbf1d, 0x69d3e1a350bcb984, 0x9382535ba5388d0f, 0x805e1591d721e13a },
                { 0x362aedb2cbb4e15f, 0xb26d4e38e2068ff5, 0x32fa2230c45f3875, 0xdbd975eb0f0b10e3, 0xd69dbc9539ca9a98, 0x3c9fe7c9da36fc18 },
            },
            a[][6] = {
                { 0x94ec878d1a4356b3, 0xdc7f34e2dd5fc042, 0x9d419b78b22bbf1d, 0x69d3e1a350bcb984, 0x9382535ba5388d0f, 0x805e1591d721e13a },
                { 0x362aedb2cbb4e15f, 0xb26d4e38e2068ff5, 0x32fa2230c45f3875, 0xdbd975eb0f0b10e3, 0xd69dbc9539ca9a98, 0x3c9fe7c9da36fc18 },
            };

        uint64_t t[6];

        for (int i=0; pass && (i<2); i++) {
            fp_cpy(t, q[i]);

            for (int j=0; j<6; j++)
                if (t[j] != a[i][j])
                    pass = false;

            if (!pass)
                printf("\n0x%016lx%016lx%016lx%016lx%016lx%016lx\n != 0x%016lx%016lx%016lx%016lx%016lx%016lx\nfp_cpy: FAIL\n",
                t[5], t[4], t[3], t[2], t[1], t[0],
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

            ++count;
        }
    }

    // fp_reduce6

    if (pass) {
        __const__ fp_t
            q[] = {
                { 0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p-1
                { 0xb9feffffffffaaab, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p
                { 0xb9feffffffffaaac, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^384-1
                { 0x8592755f85e8d727, 0x581788d31ede00fc, 0xd2908bd8c20e727c, 0x894c03a208420535, 0x3d3db3045ad56524, 0x2a318cf9cca60638 },
                { 0xccb4a5259c71ec5b, 0x8925c20877920d68, 0xe9ea6726afa1ac6a, 0x4a2e1c4163c05dca, 0x425e4a8e92fad8be, 0xdcec9f5991788284 },
                { 0x02044569c8b77a3e, 0x32973f4932540003, 0x7c9921ccff9fc38c, 0xdb84203d7e11de61, 0xa03712056f09f5b6, 0xba1104f592facb0d },
                { 0x4ddb0ff47d437f82, 0xbd2d3980fd8ff97d, 0x792a8035f135d8aa, 0xb5f66d21c0c92895, 0xa85a6dcdc58457d5, 0x755bb1f21e2fcba1 },
            },
            a[] = {
                { 0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p-1
                { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // p
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // p+1
                { 0x760900000002fffc, 0xebf4000bc40c0002, 0x5f48985753c758ba, 0x77ce585370525745, 0x5c071a97a256ec6d, 0x15f65ec3fa80e493 }, // 2^384-1
                { 0xcb93755f85e92c7c, 0x396b88d46d8a00fc, 0x6b5fb937cb5d7c58, 0x24d4b81d14bcf276, 0xf2220b4e1789b84d, 0x10307b0f93261f9d },
                { 0xfcbca5259c749703, 0x93c5c212ecf20d6a, 0xb063d21efa19fb49, 0x2673c019c797c7cf, 0xe9810cdc789d7203, 0x0ce41007c5794db1 },
                { 0xec0b4569c8b9cf91, 0x5be33f5259080004, 0xaa435f6640c9088f, 0x1c410f9ad56e5b25, 0x92757c0997f83bd3, 0x0409878e007b7cd5 },
                { 0x65df0ff47d44d4d6, 0x427d3986383ff97e, 0xdc6735b21672001a, 0x24193f0df2b4dd97, 0x7bebcef4b855a478, 0x0d576a4938303138 },
            };

        fp_t t;

        for (int i=0; pass && (i<8); i++) {
            fp_cpy(t, q[i]);

            fp_reduce6(t);

            for (int j=0; j<6; j++)
                if (t[j] != a[i][j])
                    pass = false;

            if (!pass)
                printf("\n0x%016lx%016lx%016lx%016lx%016lx%016lx\n != 0x%016lx%016lx%016lx%016lx%016lx%016lx\nfp_cpy: FAIL\n",
                t[5], t[4], t[3], t[2], t[1], t[0],
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);
            ++count;
        }
    }

    // fp_eq, fp_neq

    if (pass) {
        uint64_t
            t[][6] = {
                { 0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p-1
                { 0xb9feffffffffaaab, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p
                { 0xb9feffffffffaaac, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^384-1
            },
            u[][6] = {
                { 0x73fdffffffff5555, 0x3d57fffd62a7ffff, 0xce61a541ed61ec48, 0xc8ee9709e70a257e, 0x96374f6c869759ae, 0x340223d472ffcd34 }, // 2p-1
                { 0x2dfcffffffff0001, 0x5c03fffc13fbffff, 0x359277e2e412e26c, 0x2d65e28eda8f383e, 0xe152f722c9e30686, 0x4e0335beac7fb3ce }, // 3p
                { 0xa1fafffffffe5558, 0x995bfff976a3fffe, 0x03f41d24d174ceb4, 0xf6547998c1995dbd, 0x778a468f507a6034, 0x820559931f7f8103 }, // 5p+1
                { 0x760900000002fffc, 0xebf4000bc40c0002, 0x5f48985753c758ba, 0x77ce585370525745, 0x5c071a97a256ec6d, 0x15f65ec3fa80e493 }, // 2^384-1
            };

        for (int i=0; pass && i<4; i++) {
            uint64_t x[6];

            fp_cpy(x, t[i]);

            for (int j=0; pass && j<4; j++) {
                uint64_t y[6];

                fp_cpy(y, u[j]);

                uint64_t
                    eq  = fp_eq (x, y),
                    neq = fp_neq(x, y);

                if (eq == neq) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result, eq = %lx, neq = %lx\n", i, j, eq, neq);
                }

                if ((i == j) && !eq) {
                    pass = false;

                    printf("%d,%d: FAIL A: fp_eq claims inequality between these values:\n", i, j);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    t[i][5], t[i][4], t[i][3], t[i][2], t[i][1], t[i][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    x[5], x[4], x[3], x[2], x[1], x[0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    u[j][5], u[j][4], u[j][3], u[j][2], u[j][1], u[j][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    y[5], y[4], y[3], y[2], y[1], y[0]);

                    printf("eq = %lx, neq = %lx\n", eq, neq);
                }

                if ((i != j) && eq) {
                    pass = false;

                    printf("%d,%d: FAIL B: fp_eq claims equality between these values:\n", i, j);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    t[i][5], t[i][4], t[i][3], t[i][2], t[i][1], t[i][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    x[5], x[4], x[3], x[2], x[1], x[0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    u[j][5], u[j][4], u[j][3], u[j][2], u[j][1], u[j][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    y[5], y[4], y[3], y[2], y[1], y[0]);

                    printf("eq = %lx, neq = %lx\n", eq, neq);
                }

                if ((i == j) && neq) {
                    pass = false;

                    printf("%d,%d: FAIL C: fp_neq claims inequality between these values:\n", i, j);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    t[i][5], t[i][4], t[i][3], t[i][2], t[i][1], t[i][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    x[5], x[4], x[3], x[2], x[1], x[0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    u[j][5], u[j][4], u[j][3], u[j][2], u[j][1], u[j][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    y[5], y[4], y[3], y[2], y[1], y[0]);

                    printf("eq = %lx, neq = %lx\n", eq, neq);
                }

                if ((i != j) && !neq) {
                    pass = false;

                    printf("%d,%d: FAIL D: fp_neq claims equality between these values:\n", i, j);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    t[i][5], t[i][4], t[i][3], t[i][2], t[i][1], t[i][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    x[5], x[4], x[3], x[2], x[1], x[0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    u[j][5], u[j][4], u[j][3], u[j][2], u[j][1], u[j][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    y[5], y[4], y[3], y[2], y[1], y[0]);

                    printf("eq = %lx, neq = %lx\n", eq, neq);
                }
                ++count;
            }
        }
    }

    // fp_neg

    if (pass) {
        uint64_t
            q[][6] = {
                { 0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p-1
                { 0xb9feffffffffaaab, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p
                { 0xb9feffffffffaaac, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^384-1
                { 0x84e69d2c7894f76b, 0x4d8879222ab48ea3, 0xcd2c630dbcf4adc7, 0x4a29ccf54ce7cf87, 0x59d85eba877626b1, 0x8cc475b23946cd47 },
                { 0xb94f1b28b3dd6c63, 0x8ee855618f40fdde, 0xd022eb768da93eef, 0xba434b1f4bf1f20a, 0xa3548f2e30ddd9e5, 0x602d7f007c75ea20 },
                { 0x9112c113e0171ad3, 0xa2ddf8cdde767cff, 0xa7b2394c1e305a80, 0xd4343677f1ce5870, 0x27ebab61ee73762a, 0x85ac30859e8635ac },
                { 0xf8638fe1832a9f6f, 0x29fd21726eb7f5d9, 0x17d1f4993f9bfd51, 0xbdc07041d2317755, 0xbb40a07f7d360d4b, 0x5cf091ced52b5918 },
            },
            a[][6] = {
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // p+1
                { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // p
                { 0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p-1
                { 0x9feffffffffaaab1, 0xeabfffeb153ffffb, 0x730d2a0f6b0f6241, 0x4774b84f38512bf6, 0xb1ba7b6434bacd76, 0xa0111ea397fe69a4 }, // i16p-2^384+1
                { 0xd71362d387690897, 0x6a7f86d5fd43715a, 0x9df88cb80b311711, 0x10a1f8286836a0f4, 0x68cd8f8b0c4fe65b, 0x0f41f5cb1fb89a56 },
                { 0x2eace4d74c213e49, 0xebc7aa99360f0220, 0xcca05f0d4d1a99a0, 0xd799e2f4822258f2, 0x891a0faadc50d977, 0x07d6c8a86989b048 },
                { 0xcae73eec1fe6e52f, 0x152a072a498182fe, 0xc372b679a9f56a58, 0x86978ea5c350180b, 0x9aba42e3a55296e1, 0x165a3af7ba7931f1 },
                { 0xef98701e7cd40b3d, 0x50b2de8856980a24, 0x84f155ea9b27db3f, 0xd41cbdd1fbe2d3a8, 0x712dfe598ff8a611, 0x0b13b5da10d44150 },
            };

        fp_t t;

        for (int i=0; pass && (i<8); i++) {
            fp_neg(t, q[i]);

            if (fp_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fp_neg: FAIL\n");

                printf("- 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                q[i][5], q[i][4], q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                t[5], t[4], t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fp_x2

    if (pass) {
        fp_t
            q[] = {
                { 1, 0, 0, 0, 0, 0 }, // 1
                { 0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p-1
                { 0xb9feffffffffaaab, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p
                { 0xb9feffffffffaaac, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^384-1
                { 0xb0fd3baee62cc5f8, 0x7dce9e716b579ff6, 0xc4d51fa950015f9f, 0x993b9498361cc386, 0x5cf286851064fbc2, 0xc88b1a71b810d7b4 },
                { 0x697a35dc9725f3fa, 0x1fe337b2daa5e009, 0xfa211a6945f9fe12, 0xdc186fad1eaf837e, 0x33a1d44fb90ef709, 0x60e55c7dd5623b5c },
                { 0xdae1039a70b38fcb, 0xbcd53e042b308b8b, 0x524f0ac0e3bb7b9d, 0x05276cfe630c9247, 0xd5347f13e6e27df5, 0x2a05f2556d58ede0 },
                { 0xc2ee2ab900606d09, 0x6ddc945ea786dfdd, 0x04c99214cc172e77, 0xdb1ccaa41fa78db2, 0x47c1ec58ef0c159a, 0xe2b5ffe82724e8f5 },
            },
            a[] = {
                { 2, 0, 0, 0, 0, 0 }, // 2
                { 0x73fdffffffff5554, 0x3d57fffd62a7ffff, 0xce61a541ed61ec48, 0xc8ee9709e70a257e, 0x96374f6c869759ae, 0x340223d472ffcd34 }, // 2p-2
                { 0xe7fbfffffffeaaac, 0x7aaffffac54ffffe, 0x9cc34a83dac3d890, 0x91dd2e13ce144afd, 0x2c6e9ed90d2eb35d, 0x680447a8e5ff9a69 }, // 4p
                { 0xcff7fffffffd555a, 0xf55ffff58a9ffffd, 0x39869507b587b120, 0x23ba5c279c2895fb, 0x58dd3db21a5d66bb, 0xd0088f51cbff34d2 }, // 8p+2
                { 0x321300000006554d, 0xb93c0018d6c40005, 0x57605e0db0ddbb51, 0x8b256521ed1f9bcb, 0x6cf28d7901622c03, 0x11ebab9dbb81e28c }, // 2^385-2
                { 0x7c09775dcc5e8beb, 0x2f893cf672c33ff1, 0x7dcde7e42ba45321, 0x4f79bc66276d6dd6, 0x5346395c2f5ad6e6, 0x0b06282a11a32c5e },
                { 0xbcfb6bb92e4e3d47, 0x69126f6edbffc014, 0x21ec726bcd1d4127, 0xf8edceb794bb83c2, 0x598212a39b0c342f, 0x0bc33b9418452880 },
                { 0x87c50734e1681f95, 0x1da67c0c42651718, 0x6f0b9d9ee36414cf, 0xdce8f76deb89ec50, 0xc916070503e1f563, 0x0608aeec2e3227f2 },
                { 0x2bed557200c684b7, 0xd24d28d38879bfc0, 0x2f552779366e0488, 0x0a4d91741378dcae, 0x92adb5976611b0e8, 0x0b59cf427ccb81ab },
            };

        fp_t t;

        for (int i=0; pass && (i<9); i++) {
            fp_x2(t, q[i]);

            if (fp_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fp_x2: FAIL\n");
                printf("2 * 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                q[i][5], q[i][4], q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                t[5], t[4], t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fp_x3

    if (pass) {
        fp_t
            q[] = {
                { 1, 0, 0, 0, 0, 0 }, // 1
                { 0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p-1
                { 0xb9feffffffffaaab, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p
                { 0xb9feffffffffaaac, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, // p+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^384-1
                { 0x28fa38392f657271, 0xc544436985f63f09, 0xad831d7fec61991a, 0x78fc7321a936a1e5, 0x09c6c5b91889a015, 0x57dfda7aa31e3936 },
                { 0x6126dba413404f7f, 0xdfbd180c11ac58ae, 0x1d7c65d774b61a0f, 0x3e432db3a97d45bd, 0x677a048e37480e6f, 0xb77c61179fa1d259 },
                { 0x67d526d71fceee49, 0xa124d619a4cffd53, 0xe2193f2301ead151, 0x19a27683d6e13e1c, 0xa9ec82098b214407, 0xc30ce006eb7c626f },
                { 0xd8239146411d27b4, 0x2e46cfc4603a7e43, 0x636811703bb2ce69, 0x5696673b95c09810, 0xda56edce88b8077a, 0x336eda9dd2484c45 },
            },
            a[] = {
                { 3, 0, 0, 0, 0, 0 }, // 3
                { 0x2DFCFFFFFFFEFFFE, 0x5C03FFFC13FBFFFF, 0x359277E2E412E26C, 0x2D65E28EDA8F383E, 0xE152F722C9E30686, 0x4E0335BEAC7FB3CE }, // 3p-3
                { 0x5BF9FFFFFFFE0002, 0xB807FFF827F7FFFE, 0x6B24EFC5C825C4D8, 0x5ACBC51DB51E707C, 0xC2A5EE4593C60D0C, 0x9C066B7D58FF679D }, // 6p
                { 0x89F6FFFFFFFD0006, 0x140BFFF43BF3FFFD, 0xA0B767A8AC38A745, 0x8831A7AC8FADA8BA, 0xA3F8E5685DA91392, 0xEA09A13C057F1B6C }, // 9p+3
                { 0xEE1D00000009AA9E, 0x86840025E97C0007, 0x4F7823C40DF41DE8, 0x9E7C71F069ECE051, 0x7DDE005A606D6B99, 0x0DE0F8777C82E085 }, // 3*2^384-3
                { 0x36F8A8AB8E33ACA5, 0x1D14CA49A49ABD1E, 0x00A11E36223B2DE7, 0x7E4C663378712A37, 0x2E3FC40CA8A81FD6, 0x0394DC49AA5BA99B },
                { 0xE18992EC39C7EE76, 0x1B1B483FA9210A10, 0xE173EA52219E1D39, 0x7D005733028D4783, 0x0D294BB720A2FDA2, 0x045EAB1027678C64 },
                { 0x3B9574855F742029, 0x40A68269B137F800, 0xC819A393D28B4CDA, 0xAAA6E61E97341DE3, 0x89651C72D8E2F192, 0x0D0F15F3D177560C },
                { 0xE66FB3D2C35921C5, 0xF1786F53AA0B7ACC, 0x2644172BE1A39C86, 0x0D6EBC19FFA86A74, 0x177A82DC49ADB63A, 0x18473646575963CE },
            };

        fp_t t;

        for (int i=0; pass && (i<9); i++) {
            fp_x3(t, q[i]);

            if (fp_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fp_x3: FAIL\n");
                printf("3 * 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                q[i][5], q[i][4], q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                t[5], t[4], t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fp_add

    if (pass) {
        fp_t
            x[] = {
                { 0xf9eb49b8c7a56a91, 0x0ef126fc4e085685, 0x205eeb7c38dbbf9d, 0xf39db86dd74ba55e, 0xb53be4eb9f0aba5b, 0xa9adcd7739bd74cf },
                { 0xfe2acbd68b1dab4c, 0xc887627968616adb, 0xebc3269a81507575, 0x0712c334b66a5db8, 0x2f2fb5eebf0bbbe7, 0x1228a231f10f3969 },
                { 0xb5d627d5148d5ee4, 0x9f90924cf9a20f59, 0x44583bcadb79bbb9, 0x5a9d273d378355e4, 0x07e79b33160b873e, 0xb600a76ec90fb9ed },
                { 0xc08d788a248344e9, 0x03a80291c374aa79, 0xff4b832001a36d0d, 0xe9c922679104943d, 0xbb06a2547d248706, 0x504eb835fcb63c60 },
                { 0x67cca4f1fb53d0ac, 0xd105c0e0e18acc1e, 0xcab450c115f31533, 0x3566711770d0337e, 0xaed37a617a390c6a, 0xe31f46556875472c },
                { 0x2b439aa3af0bac40, 0x3feee451e8ead810, 0xc016a5c51c5a6199, 0x24fc2d78e8c1dd93, 0x58cadad8d1961158, 0xccc5860e7b21e1c6 },
                { 0xcc2acd2768c764cf, 0x5dc0ecf1246274c2, 0x795aa4384f5ab137, 0xbff1464b3d3d6939, 0x4c2caf2b9b11fa95, 0x1f5ceaaf4f7d6de4 },
                { 0x10e18a590b63d8e6, 0x2b19856f169e090c, 0x05dfc520280b525e, 0xea0f7df125f638d4, 0x068815eb1cb349ca, 0xf74a8f42e7f91926 },
            },
            y[] = {
                { 0x517cb18f6844fe73, 0xbf8c5be950010a8e, 0x579c2cab14ba5662, 0x81c14702fd352d79, 0x620cdaf477088c3e, 0xc4fec1d957afa21a },
                { 0x4a8cd70deb76aebb, 0x9e7a6dec7050c6e4, 0x8a6620ebfcebd51b, 0x66e5a460cc82bc8c, 0x95247c6c292eafce, 0x49031305fd08b422 },
                { 0x544ba69e20b1ad57, 0x00f67b5f1c5b61c7, 0x719912e5a5aba394, 0x33ccff2766b4f4c9, 0xf2ea148ea21f8495, 0xd7a40e39d1285b4b },
                { 0xb3bdb711a08c7275, 0xfb82f5c9bad4ec79, 0x50a27f38bf3302ca, 0x19ac2bbd2a9df826, 0xe57584bd59781e9a, 0xb0a9f4c1bd067f9e },
                { 0x0c6e44f0c9175538, 0x2c9e5523543c95a1, 0x0cab9a0b906628af, 0xa1bb974e40d1e8df, 0x00b44b033aff2e55, 0x6ce53032663eb897 },
                { 0xb780e61fd6be4d13, 0x767e8edccfd2e850, 0x797cd2e975ffcc1d, 0x81cfbe199c2e992a, 0x8c36e257243dfcc3, 0x134aa1d8969bbf42 },
                { 0xc7c641fafcc5c79a, 0xe33e72638d433936, 0x900e218d7c6fa297, 0x3196afb585723831, 0x364422505bd94d9e, 0x7db61267bd4d87f2 },
                { 0x0a287d49e03b3a16, 0x7885df1173b1f516, 0xdf54afe9bc80f448, 0x0087ad3b21681d47, 0x5e6809f0b1c9659f, 0xa801fa7b481d5cca },
            },
            a[] = {
                { 0x1f75fb482fef13aa, 0x211582f7eb716118, 0xd34f9359cfe8a006, 0xf6d8de2b8339cc5f, 0xfbc593e867efd2d2, 0x029d94816c6e7a79 },
                { 0x1abaa2e476955a06, 0x0afdd069c4b631c1, 0x4096cfa39a296825, 0x40928506a85de207, 0xe3013b381e57652f, 0x0d287f79419839bc },
                { 0x2430ce7335440c36, 0xd4730dbfb2117125, 0xaa14f7420cc6f32f, 0xab6cb99a596c3176, 0x9432dc13c6bbeb34, 0x0794a8ef3bb9922e },
                { 0xea542f9bc512b75b, 0xeb1ef867425596f5, 0xaf369ab0149dc892, 0x7b43a6782bf4e3a9, 0xfc8341a978f3920e, 0x16ef0bbbb43da092 },
                { 0xbc46e9e2c46f25e0, 0x8d941613e5d761c2, 0x01160b41160db431, 0x218a7e2a47653b65, 0x2a3be8d98dac20a7, 0x17f79f8d1cb53088 },
                { 0x12cc80c385cca3fb, 0xc10d73392e1dc063, 0x000ce3a6dcd27c95, 0x83118f6ae8c7e0c3, 0x8c247f7ddb76a760, 0x1007989545be6c36 },
                { 0x37f70f22658f2c67, 0x88f75f5c89adadfb, 0x9e43d60003a48ef6, 0x96bc30e30d9130ee, 0xbfcae33663253b27, 0x010c9199b3cb8e38 },
                { 0x351907a2eba412f7, 0xd78b64942663fe26, 0xd9581d9b702dda88, 0x0799be6202923ce4, 0xfe514c2ddd0d8ecb, 0x193c7d04d197f2e5 },
            };

        fp_t t;

        for (int i=0; pass && (i<8); i++) {
            fp_cpy(t, x[i]);

            fp_add(t, t, y[i]);

            if (fp_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fp_add: FAIL\n");
                printf("0x%016lx%016lx%016lx%016lx%016lx%016lx + 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                x[i][5], x[i][4], x[i][3], x[i][2], x[i][1], x[i][0],
                y[i][5], y[i][4], y[i][3], y[i][2], y[i][1], y[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                t[5], t[4], t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fp_sub

    if (pass) {
        uint64_t
            x[][6] = {
                { 0x0606c98eee788f23, 0xe14fb16a3e46bc1a, 0x5cd48d8914135670, 0x2571278e41d8d960, 0x5938348c63e5ee03, 0x79c2dcd6c072a0eb },
                { 0x8067d3e25df5a479, 0x7b1b315337ca38d3, 0x2c7ec7596a88e143, 0x6257ac6e44173664, 0x496ce21047d6852e, 0xdd0e7887bd1fae7d },
                { 0x9a0258c11328fe77, 0xc73113f9f74960a1, 0xde8c9d728a65a66b, 0xdd5ac0fc26829b45, 0x3544b4a08022241b, 0x4d1ae12876563e2f },
                { 0xd017f19b53279294, 0xb3462f2017a23e85, 0x4549a8ae17749941, 0x15c5d82b5e0d969b, 0x2bf8fc83ec477308, 0xe42f83194305299d },
                { 0xc74a02b87420007d, 0x382f4bdd58a901fa, 0x0aa57ffbe3f4eda8, 0xce9aeb581a6f2561, 0xa62b1cef87f16d08, 0xcf43cbed7acb5065 },
                { 0x4a096555e53dfe92, 0x92f4894cf1f8e6a2, 0x76fb671eeccdab6f, 0x3ce8377fe91310d9, 0xf28e9ae385c7b25e, 0xd76412152770edf7 },
                { 0x505fd2ed275e299f, 0xde174f19f1545aee, 0x9932233eb0565fcd, 0xde5d2875e048c920, 0x1153b798996a5f4c, 0xe69ee87aff35023d },
                { 0x0045c3a6e97902c7, 0x42d34a4cd6288347, 0xe5096ba7b364ef06, 0xc4873604dbd43251, 0x803049704c5e877a, 0xa9d696814e42f2d7 },
            },
            y[][6] = {
                { 0x5c7f1f3c4dbe87e5, 0xf3e1e2121586270a, 0x6adb7627e9f6aeef, 0x01d9e767cde7761d, 0x5676c850dc0abe6a, 0x0525d4660db16d22 },
                { 0xa4569ec7699b9089, 0x8bb7ced6eaee19c9, 0x500f5f5b67ab1495, 0x4e6c6c189328a37c, 0x4726bb54e06ddf10, 0x00b82535d67c3d30 },
                { 0x8f292c3829819028, 0x1f703fbc7fb74560, 0xb56748bca526c301, 0x402323c56b47ff4d, 0x1f46f673e00873a1, 0x1a85df184987b062 },
                { 0xe4ce8660242014ba, 0x97ce65d293c7de4e, 0x69ad05e1728edb01, 0x1170a2b515955e6e, 0xa015fddb814bb5cc, 0xf2adc10711825b80 },
                { 0xcc0eb15e14de85f1, 0x396659fe6d298d52, 0x15225553b867d6ae, 0xe7a0b5a0290501f7, 0x9fee9451065ba006, 0x159ad40623ff267e },
                { 0xc594898ebfaab0b8, 0x192ba26ed0ba4476, 0x7c27cf5b5ea73d45, 0x2c3994f637e68acb, 0x02a555b27cc700f8, 0x2706fb06c7aa6448 },
                { 0xfaca28683dee120e, 0x6507d4925dbb9cd8, 0xccbf478a6c2b8bec, 0x17ed638bbc19695b, 0x9f3f454d588f659b, 0x05aa619987706308 },
                { 0x7b892f0f831ec8d8, 0xb21857d01de5351a, 0x85f8ca724eb03970, 0x280a417b3dcd44d1, 0xc8ea30df2b736be2, 0x1e827c98a11f5038 },
            },
            a[][6] = {
                { 0xc18baa52a0bb5c92, 0x72bdcf5d63709510, 0x5535ccdd4f58cef0, 0x91ba1212a5dd1845, 0xd652cd627aac7c3b, 0x0c98c0c7ccc1995f },
                { 0x0c19351af45cbe98, 0xfa036286c23c1f0c, 0xa2e8d2f64d561b8c, 0xf030e42e14c5fcec, 0xa968e9094d0b3f62, 0x0c4dc4001aa43c7a },
                { 0x50da2c88e9a7c3a4, 0x8914d43ec63e1b41, 0xc1f48214ee8ded46, 0x38c051b1c7b58938, 0xcae216765cce03a3, 0x1893f025f34ea732 },
                { 0xa5486b3b2f072885, 0x3a23c94c352e6036, 0x42cd756d9b96b464, 0x68cc80fb3bfd4aec, 0xd6fea65eae476a13, 0x0b82d3fc6b02b4b6 },
                { 0xe542515a5f43cfdf, 0x2814f1e8123374a9, 0x232d68416cb65bfd, 0x27b7251548c6a02e, 0xf87af2a2aa84131e, 0x03a17a7fc44cdbae },
                { 0x287adbc725954dd8, 0xc1c0e6e5f946a22d, 0x8faea7fdc600a951, 0xb5e2dd6bfc0e1591, 0x2d4356eb753aa459, 0x1456ab9106c72212 },
                { 0x859daa84e972c239, 0x83af7a9208f8be17, 0x92ec46ac8ea322c0, 0xa2b568c28806c9c9, 0x19373499267d92f6, 0x10ebf78fabc56a62 },
                { 0xe2c19497665be498, 0xf75ef283419f4e2d, 0x5b1c8410933fe6e0, 0xa6287af0dc6d8fc3, 0x3fbbd201d070bb63, 0x094ec0558da4219b },
            };

        uint64_t t[6];

        for (int i=0; pass && (i<8); i++) {
            fp_cpy(t, x[i]);

            fp_sub(t, t, y[i]);

            if (fp_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fp_sub: FAIL\n");
                printf("0x%016lx%016lx%016lx%016lx%016lx%016lx - 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                x[i][5], x[i][4], x[i][3], x[i][2], x[i][1], x[i][0],
                y[i][5], y[i][4], y[i][3], y[i][2], y[i][1], y[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                t[5], t[4], t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fp_sqr

    if (pass) {
        uint64_t
            q[][6] = {
                { 0xcaf99e97cfdd2a55, 0x2a309e540031bb5b, 0x5d142f76332013e5, 0x0b5170efac5b0908, 0x2d5a18614d5d87dc, 0x8a28bc7a23b8ca27 },
                { 0x2eaefeba75a027dc, 0x2b658a0ce546cc14, 0x24a6e8f76c1c85af, 0xc9041e4e1182ee8e, 0x3f8e1db0ff84a316, 0x9e34a7d7bfd03f35 },
                { 0xe232fcf1e3ea21c4, 0xbd9edece603cd7ce, 0x25407371bb909e31, 0x4f7ef565a60c3557, 0x8e0c0ea2e6efeccc, 0x023f831fe9692b89 },
                { 0x61bd5a835ba24266, 0xee798cb747280467, 0x4d0458e170dbdd88, 0x2877a95e76ce586d, 0xeeeb86b0e74a3539, 0x03f9bd4ad96916c6 },
                { 0xe461923ca066ebb4, 0x39f6f7ccda29a87a, 0x4d30d5d0b742a756, 0xc91d64d567eeeb7a, 0x2d5ef5cbd3296ec9, 0x7cd618ac00921af4 },
                { 0xe21d7f15473b63b1, 0x312736f824e8b020, 0xef33e5777473b021, 0x20e59d0cd3431333, 0x242abddbb631b4d1, 0x121e682e451a272c },
                { 0xfd776f1cc48a3d25, 0x2a66488b0fb2643b, 0x16c0e36fc3e8a70a, 0xaabeba2bc6b6700c, 0x81248cd6767199a1, 0xee35cf214032be18 },
                { 0x789cdd11500578d3, 0xa9111fdacc5cbb39, 0x4f4034af157ec725, 0xe92045e05b0b1ae1, 0xe7f1740cb372b6b8, 0x2e0a1d0a32832d8e },
            },
            a[][6] = {
                { 0x1f8b0acedc008e08, 0x4a75446e7ddb973f, 0xcc2d9b595c27a3c3, 0x2233591dbd102e3b, 0xcb91dc2ff393dec9, 0x056a6de4c898189f },
                { 0xaf4cb53437bf28d1, 0xe81245eb012f43cc, 0xbeeaceb0a64cb4c9, 0xca674ff0add0f7ee, 0x1442c5aacae669a6, 0x0dfebf9a180392ce },
                { 0xfd79f6b6ff1b50dd, 0xe862843dadb66122, 0xf8ff7b7735f106be, 0xce6f5de3616d2ebd, 0xe11ff68500f8e620, 0x061270e7875966eb },
                { 0xca412b95b7a5fd9f, 0x84684c05965c1406, 0x820d0d6fb6f21711, 0x3f99a0a44883e878, 0xdfa288890d4e6269, 0x0626f98f15903db1 },
                { 0xf824534e5343cd50, 0x23ab91343d2cf826, 0xd2ac5ed48ec3fd14, 0x6e244bc004fc0aa8, 0x32897516b9481434, 0x15e74f8d7437d704 },
                { 0x5c9cc6f25d668d0e, 0x1693a22408eaad6f, 0x6bed72a49768914c, 0xaffcc66ae89de36b, 0xf2a69b042b20be6c, 0x058ad6eac88b9bb1 },
                { 0x19cce2e856a74dc8, 0x1da8d116804f9d4c, 0xdd93009a50efe467, 0x3ce8de2635b00d9d, 0x5719abbc1d70246c, 0x0d55761a05380a87 },
                { 0xcd07fb540a8570bc, 0x60f82168ad689e4a, 0xb0c7f9ac3d394ce3, 0xa558401dfa315d8f, 0xa2dbbc76eaeb3da3, 0x0996d53b8bac8ef4 },
            };

        uint64_t t[6];

        for (int i=0; pass && (i<8); i++) {
            fp_cpy(t, q[i]);

            fp_sqr(t, t);

            if (fp_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fp_sqr: FAIL\n");
                printf("0x%016lx%016lx%016lx%016lx%016lx%016lx^2\n",
                q[i][5], q[i][4], q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                t[5], t[4], t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fp_mul

    if (pass) {
        uint64_t
            x[][6] = {
                { 0x8b910eba598bcca0, 0x83118df0400ac3b1, 0x0c0aeeaacd9fdd46, 0x945bea4e11648d71, 0x8697e87a732713a0, 0x3b44a96c59c690bb },
                { 0xd92e1c921e839a93, 0x3e5d352e7af97108, 0xcc59b69c6a11800e, 0x98262bc012006f6b, 0x8fcbf94cc47a7712, 0x3b68c9ecd7f475dc },
                { 0xe846857b667502de, 0xa107f1c7aad9124e, 0xabf34c7ef950991a, 0xb1b9ad274c9a1565, 0xbc79f3857c2d3e80, 0x1c11a0b514009531 },
                { 0xd1f7fad7b0844e6e, 0xfa1de1e0e652e156, 0x9eb4079889d23a2b, 0xc887cec9b33c9221, 0xbd81e570393678b6, 0x53b0394c29c2cced },
                { 0x6d1a0016d090c7cd, 0xafe2e3d85d19e96b, 0xa94fe04dd3633195, 0x421e12948915e8cb, 0x4d376cdcc50075ea, 0x161eb7e58a70f1be },
                { 0xd26ab4330dad64e6, 0x733bba2150d75189, 0x55ad168e639da67c, 0xb37708d4e7c0695a, 0x89a5032329875fc0, 0x7796b5a9ce8ccc50 },
                { 0xaec5cd9f6d016817, 0xd038c9ba48f32b19, 0x0c4c38939953c602, 0x7926637820bdf363, 0x244b84b1405dff5b, 0x5182e3055754826c },
                { 0x40a48c01655c0ebb, 0xdb1274ff16a1c736, 0x68504f715af9afe4, 0xebb314273a513815, 0x05923cbbec717106, 0x57c3d05787a089e4 },
            },
            y[][6] = {
                { 0xc5b06f31f52da2c2, 0x22853113e7f0ce41, 0x2b26e44aea71c03d, 0x3c01f18e78a68d32, 0x53166475e534bae2, 0xa5acba7322aca411 },
                { 0xcb59275fd834faa7, 0xcf0ff42f805780af, 0xf3892c2f5a40184b, 0x6f45d1101c20d7c8, 0xed40e399d16d6bad, 0x390a4072d4249068 },
                { 0x1a4695b02139bc86, 0x385bd9c0c571d264, 0x0378847e6b358f44, 0x40a28d850e1b986b, 0x8fb29be985c830d0, 0x772b7b7c9b803667 },
                { 0xe51d09d336a1435d, 0x247edb3da18bd6cb, 0xe007639449409917, 0x3558e57591312120, 0x637f375b0fc38465, 0x4170f9de04091161 },
                { 0xf9177d062a7f9043, 0xa6ffecb2ddb82845, 0xf8a3e25080ca559d, 0x0de4ea6a4912155d, 0x5c62c18fb9f86579, 0xf25e79d416c742cf },
                { 0xcf182cc753c3c520, 0x5059a08a175ff5dd, 0xecfd9c6f82a56ce5, 0xd645c5c990f5fcbd, 0x52ad47b58cc1fd03, 0xb8885f99f0bf3dee },
                { 0xc3a3fc14c3f8daed, 0xdb1bfa5af35dc7a1, 0x5a0a27df527959df, 0x990a16f11f2fdf81, 0xb0bf9f7b46b905c8, 0xa8580fe78a036375 },
                { 0x4cdb8ce296a2d6a0, 0xb0f6f4a8be6e9801, 0x534a17e34d281cd5, 0x86fcdbc0577d64e0, 0xc274e53e00b19348, 0x8bc3d4b107836984 },
            },
            a[][6] = {
                { 0x747862c5a1198191, 0x33b89375962c7602, 0x3a4979e8e0aab70b, 0x2e378ca7519441af, 0x4e1099f1b8786e67, 0x05e7c62e5b43e436 },
                { 0xf838a86a41140dc4, 0x1af29f47de3bed28, 0x23d5a14aa7e0a73c, 0x75f08b2c0e82bfb9, 0x600abe76feee5f1e, 0x0217b1943448f6d6 },
                { 0x0864e17beb081567, 0x38df65e9926f717d, 0xa82886a9de085ae3, 0x1375d5c2c6d77f1e, 0x79da5b88b04d30b8, 0x15651d5391d8fbc1 },
                { 0xdcccb1669727711b, 0xe0723c4efe4c8d5f, 0x94f53936fea3c80a, 0xc6a266801ea98a44, 0x9ae9c72d091357d1, 0x1123fbf08b8767a9 },
                { 0xb3ebe3f607d873de, 0x8d5ef2ff067860bf, 0x2e945be8839b6a2f, 0x3c5653fe327d5be2, 0xf929fed484c449e4, 0x18eb937f76386111 },
                { 0xcc9f5aec9d3ac5ea, 0x5ba778d43d02dcb6, 0x4ea92836e8412fd6, 0x8b41748ef730efba, 0xbbbdb4026b9dd9f7, 0x0170e0a7a769721e },
                { 0x9cd24eaa0519b596, 0xa60c87db0f26a7cc, 0xf11952a94cac8fd1, 0xd646e4754ab1405c, 0xb57fa0568eaa7f7f, 0x11b50f1e687b90d4 },
                { 0x5f939b44f29e11d6, 0xcf44a5cff9efcb4a, 0x6659ba0622225dd6, 0xf39d2be3fac1a543, 0x8518de69b37ad17f, 0x07ba6ec3062b23d3 },
            };

        uint64_t t[6];

        for (int i=0; pass && (i<8); i++) {
            fp_cpy(t, x[i]);

            fp_mul(t, t, y[i]);

            if (fp_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fp_mul: FAIL\n");
                printf("0x%016lx%016lx%016lx%016lx%016lx%016lx * 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                x[i][5], x[i][4], x[i][3], x[i][2], x[i][1], x[i][0],
                y[i][5], y[i][4], y[i][3], y[i][2], y[i][1], y[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                t[5], t[4], t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fp_inv

    if (pass) {
        fp_t
            q[] = {
                { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000002, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000003, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0xa2d3a615f9856758, 0x75c36bfc647776ad, 0x33344fab14e8ed3e, 0xc810db766a4a9105, 0xef5d7270e6940fb1, 0xa127da24753d1ea9 },
                { 0x190d4848fcc4e00c, 0x3bead693b0821fce, 0x519d20d7a485985d, 0x9ad516f2aa1dbd56, 0x060d28a6bf60b67e, 0x56131b9360c13570 },
                { 0x49da829f778038d1, 0x854634fbd4c444f0, 0x664a986da9e91629, 0x46b68ee421699ad4, 0x4db2fe4cec35a193, 0x1a89fc6e8c94daed },
                { 0x9e9ab0f689270b06, 0xd6b509821b199072, 0x5039948ba7d60a02, 0x3cdef4136ccab1d9, 0xca9b6bd2f2c0d033, 0x7e26ae14d79bcb20 },
            },
            a[] = {
                { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0xDCFF7FFFFFFFD556, 0x0F55FFFF58A9FFFF, 0xB39869507B587B12, 0xB23BA5C279C2895F, 0x258DD3DB21A5D66B, 0x0D0088F51CBFF34D },
                { 0x26A9FFFFFFFFC71D, 0x1472AAA9CB8D5555, 0x9A208C6B4F20A418, 0x984F87ADF7AE0C7F, 0x32126FCED787C88F, 0x11560BF17BAA99BC },
                { 0xC77B8501E2F4FC46, 0x35FC297BA0FAFCAA, 0x02A111064E34921E, 0x8C9AD0D15A2820BB, 0xCF68F8F27B8EDE3C, 0x0E9C67AFABF5D493 },
                { 0x6D31E7BC1D389137, 0x2EB64F358273A506, 0x64D8FCD87FB339CF, 0x3BFF350FA7A07E6A, 0x9A6F77ABB4F0F685, 0x17465FC34A55F3C9 },
                { 0xB362B4066CF7E727, 0x352A8805C8D7482F, 0x471835C5660FF87D, 0xF890D4921C97BA88, 0xE4C5A766B1E9A79C, 0x10F7778268F6B55F },
                { 0x5FB388A220997B22, 0x6FDD9EEF36C6EF99, 0x2B0E3C9B46BEE64D, 0xFBAE3CCB5C9D6AE7, 0xFFFCB6D3F89BB725, 0x0EC41F3C16BB37FF },
            };

        fp_t t;

        for (int i=0; pass && (i<8); i++) {
            fp_inv(t, q[i]);

            for (int j=0; j<6; j++)
                if (t[j] != a[i][j])
                    pass = false;

            if (!pass) {
                printf("fp_inv: FAIL\n");

                printf("1 / 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                q[i][5], q[i][4], q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                t[5], t[4], t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // Mandelbrot iteration

    if (pass) {
        fp_t
            q[] = {
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000002, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000003, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x7bf6baf3e15212b9, 0x33b894f547f36037, 0xf63c05923a4ece1a, 0x743a6fad39e26e69, 0xbd01dd2df2801c98, 0x13cb2cac4de43f03 },
            },
            a[] = {
                { 0x7bf6baf3e15212b9, 0x33b894f547f36037, 0xf63c05923a4ece1a, 0x743a6fad39e26e69, 0xbd01dd2df2801c98, 0x13cb2cac4de43f03 },
                { 0x229f7dbddec29029, 0xc967d5d7e7a566d9, 0x8035156b65fc81ae, 0xc7589446725c23c8, 0x9ff2cc573669de3d, 0x12c0ec4bf3d41dd5 },
                { 0x9a0d2e51810d7c67, 0x9cdffe69a269185b, 0x979132a177105382, 0x10f78c01a49355d5, 0x41982e9bbf0f18e6, 0x160db6f5b80d799f },
                { 0xb0a6fa474fcd5ddd, 0x133bedb13c02a6bd, 0xf95a8cc2cc516a9d, 0xc81d7d618c6f179c, 0x2b965f0a7f364f32, 0x17f75e634dc09ff8 },
            };

        fp_t z, x;

        for (int i=0; pass && (i<4); i++) {
            fp_zero(z);
            fp_cpy(x, q[i]);

            for (int j=0; j<10000; j++) {
                fp_sqr(z, z);
                fp_add(z, z, x);
            }

            if (fp_neq(z, a[i])) {
                pass = false;

                printf("fp mandelbrot: FAIL %d\n", i);

                printf("0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                q[i][5], q[i][4], q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                a[i][5], a[i][4], a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx%016lx%016lx\n",
                z[5], z[4], z[3], z[2], z[1], z[0]);
            }

            ++count;
        }
    }

    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
