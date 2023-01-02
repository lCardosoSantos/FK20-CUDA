// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"
#include "frtest.cuh"

__global__ void FrTestKAT(testval_t *) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    // fr_cpy

    if (pass) {
        __const__ fr_t
            q[] = {
                { 0x9d419b78b22bbf1d, 0x69d3e1a350bcb984, 0x9382535ba5388d0f, 0x805e1591d721e13a },
                { 0x32fa2230c45f3875, 0xdbd975eb0f0b10e3, 0xd69dbc9539ca9a98, 0x3c9fe7c9da36fc18 },
            },
            a[] = {
                { 0x9d419b78b22bbf1d, 0x69d3e1a350bcb984, 0x9382535ba5388d0f, 0x805e1591d721e13a },
                { 0x32fa2230c45f3875, 0xdbd975eb0f0b10e3, 0xd69dbc9539ca9a98, 0x3c9fe7c9da36fc18 },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<2); i++) {
            fr_cpy(t, q[i]);

            for (int j=0; j<4; j++)
                if (t[j] != a[i][j])
                    pass = false;

            if (!pass)
                printf("\n0x%016lx%016lx%016lx%016lx\n != 0x%016lx%016lx%016lx%016lx\nfr_cpy: FAIL\n",
                t[3], t[2], t[1], t[0],
                a[i][3], a[i][2], a[i][1], a[i][0]);

            ++count;
        }
    }

    // fr_reduce4

    if (pass) {
        __const__ fr_t
            q[] = {
                { 0xFFFFFFFF00000000, 0x53BDA402FFFE5BFE, 0x3339D80809A1D805, 0x73EDA753299D7D48 }, // r-1
                { 0xFFFFFFFF00000001, 0x53BDA402FFFE5BFE, 0x3339D80809A1D805, 0x73EDA753299D7D48 }, // r
                { 0xFFFFFFFF00000002, 0x53BDA402FFFE5BFE, 0x3339D80809A1D805, 0x73EDA753299D7D48 }, // r+1
                { 0xfffffffe00000001, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2r-1
                { 0xfffffffe00000002, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2r
                { 0xfffffffe00000003, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2r+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^256-1
                { 0xd2908bd8c20e727c, 0x894c03a208420535, 0x3d3db3045ad56524, 0x2a318cf9cca60638 },
                { 0xe9ea6726afa1ac6a, 0x4a2e1c4163c05dca, 0x425e4a8e92fad8be, 0xdcec9f5991788284 },
                { 0x7c9921ccff9fc38c, 0xdb84203d7e11de61, 0xa03712056f09f5b6, 0xba1104f592facb0d },
                { 0x792a8035f135d8aa, 0xb5f66d21c0c92895, 0xa85a6dcdc58457d5, 0x755bb1f21e2fcba1 },
            },
            a[] = {
                { 0xFFFFFFFF00000000, 0x53BDA402FFFE5BFE, 0x3339D80809A1D805, 0x73EDA753299D7D48 }, // r-1
                { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // r
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // r+1
                { 0xFFFFFFFF00000000, 0x53BDA402FFFE5BFE, 0x3339D80809A1D805, 0x73EDA753299D7D48 }, // 2r-1
                { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // 2r
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // 2r+1
                { 0x00000001fffffffd, 0x5884b7fa00034802, 0x998c4fefecbc4ff5, 0x1824b159acc5056f }, // 2^256-1
                { 0xd2908bd8c20e727c, 0x894c03a208420535, 0x3d3db3045ad56524, 0x2a318cf9cca60638 },
                { 0xe9ea6727afa1ac69, 0xf670783e63c201cb, 0x0f247286895900b8, 0x68fef80667db053c },
                { 0x7c9921cdff9fc38b, 0x87c67c3a7e138262, 0x6cfd39fd65681db1, 0x46235da2695d4dc5 },
                { 0x792a8036f135d8a9, 0x6238c91ec0cacc96, 0x752095c5bbe27fd0, 0x016e0a9ef4924e59 },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<11); i++) {
            fr_cpy(t, q[i]);

            fr_reduce4(t);

            for (int j=0; j<4; j++)
                if (t[j] != a[i][j])
                    pass = false;

            if (!pass)
                printf("\n0x%016lx%016lx%016lx%016lx\n != 0x%016lx%016lx%016lx%016lx\nfr_cpy: FAIL\n",
                t[3], t[2], t[1], t[0],
                a[i][3], a[i][2], a[i][1], a[i][0]);
            ++count;
        }
    }

    // fr_eq, fr_neq

    if (pass) {
        fr_t
            t[] = {
                { 0xFFFFFFFF00000000, 0x53BDA402FFFE5BFE, 0x3339D80809A1D805, 0x73EDA753299D7D48 }, // r-1
                { 0xFFFFFFFF00000001, 0x53BDA402FFFE5BFE, 0x3339D80809A1D805, 0x73EDA753299D7D48 }, // r
                { 0xFFFFFFFF00000002, 0x53BDA402FFFE5BFE, 0x3339D80809A1D805, 0x73EDA753299D7D48 }, // r+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^256-1
            },
            u[] = {
                { 0xfffffffe00000001, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2r-1
                { 0xfffffffe00000002, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2r
                { 0xfffffffe00000003, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2r+1
                { 0x00000001fffffffd, 0x5884b7fa00034802, 0x998c4fefecbc4ff5, 0x1824b159acc5056f }, // 2^256-1
            };

        for (int i=0; pass && i<4; i++) {
            __shared__ fr_t x;

            fr_cpy(x, t[i]);

            for (int j=0; pass && j<4; j++) {
                __shared__ fr_t y;

                fr_cpy(y, u[j]);

                int
                    eq  = fr_eq (x, y),
                    neq = fr_neq(x, y);

                if (eq == neq) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result, eq = %x, neq = %x\n", i, j, eq, neq);
                }

                if ((i == j) && !eq) {
                    pass = false;

                    printf("%d,%d: FAIL A: fr_eq claims inequality between these values:\n", i, j);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    t[i][5], t[i][4], t[i][3], t[i][2], t[i][1], t[i][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    x[5], x[4], x[3], x[2], x[1], x[0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    u[j][5], u[j][4], u[j][3], u[j][2], u[j][1], u[j][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    y[5], y[4], y[3], y[2], y[1], y[0]);

                    printf("eq = %x, neq = %x\n", eq, neq);
                }

                if ((i != j) && eq) {
                    pass = false;

                    printf("%d,%d: FAIL B: fr_eq claims equality between these values:\n", i, j);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    t[i][5], t[i][4], t[i][3], t[i][2], t[i][1], t[i][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    x[5], x[4], x[3], x[2], x[1], x[0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    u[j][5], u[j][4], u[j][3], u[j][2], u[j][1], u[j][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    y[5], y[4], y[3], y[2], y[1], y[0]);

                    printf("eq = %x, neq = %x\n", eq, neq);
                }

                if ((i == j) && neq) {
                    pass = false;

                    printf("%d,%d: FAIL C: fr_neq claims inequality between these values:\n", i, j);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    t[i][5], t[i][4], t[i][3], t[i][2], t[i][1], t[i][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    x[5], x[4], x[3], x[2], x[1], x[0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    u[j][5], u[j][4], u[j][3], u[j][2], u[j][1], u[j][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    y[5], y[4], y[3], y[2], y[1], y[0]);

                    printf("eq = %x, neq = %x\n", eq, neq);
                }

                if ((i != j) && !neq) {
                    pass = false;

                    printf("%d,%d: FAIL D: fr_neq claims equality between these values:\n", i, j);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    t[i][5], t[i][4], t[i][3], t[i][2], t[i][1], t[i][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    x[5], x[4], x[3], x[2], x[1], x[0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                    u[j][5], u[j][4], u[j][3], u[j][2], u[j][1], u[j][0]);

                    printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                    y[5], y[4], y[3], y[2], y[1], y[0]);

                    printf("eq = %x, neq = %x\n", eq, neq);
                }
                ++count;
            }
        }
    }

    // fr_neg

    if (pass) {
        fr_t
            q[] = {
                { 0xfffffffe00000001, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2p-1
                { 0xfffffffe00000002, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2p
                { 0xfffffffe00000003, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2p+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^256-1
                { 0x8cc475b23946cd47, 0x59d85eba877626b1, 0x4a29ccf54ce7cf87, 0xcd2c630dbcf4adc7 },
                { 0x602d7f007c75ea20, 0xa3548f2e30ddd9e5, 0xba434b1f4bf1f20a, 0xd022eb768da93eef },
                { 0x85ac30859e8635ac, 0x27ebab61ee73762a, 0xd4343677f1ce5870, 0xa7b2394c1e305a80 },
                { 0x5cf091ced52b5918, 0xbb40a07f7d360d4b, 0xbdc07041d2317755, 0x17d1f4993f9bfd51 },
            },
            a[] = {
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // r+1
                { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 }, // r
                { 0xffffffff00000000, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48 }, // r-1
                { 0xfffffffc00000005, 0x4ef6900bfff96ffb, 0xcce7602026876015, 0xcfb69d4ca675f520 }, // 3p-2^256+1
                { 0x733b8a4bc6b932bb, 0x4da2e94b7886914c, 0x1c49e31ac65be083, 0x1aaeeb9896464cc9 },
                { 0x9fd280fd838a15e2, 0x0426b8d7cf1ede18, 0xac3064f0c751be00, 0x17b8632fc591bba0 },
                { 0x7a53cf786179ca56, 0x7f8f9ca4118941d3, 0x923f79982175579a, 0x4029155a350aa00f },
                { 0xa30f6e302ad4a6e9, 0x987d038382c84eb3, 0x757967c6377060af, 0x5c1bb2b9ea017ff6 },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<8); i++) {
            fr_cpy(t, q[i]);

            fr_neg(t);

            if (fr_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fr_neg: FAIL\n");

                printf("%d: 0x%016lx%016lx%016lx%016lx\n", i,
                q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx\n",
                a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx\n",
                t[3], t[2], t[1], t[0]);
        }

            ++count;
        }
    }

    // fr_x2

    if (pass) {
        fr_t
            q[] = {
                { 0xffffffff00000000, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48 }, // r-1
                { 0xffffffff00000001, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48 }, // r
                { 0xffffffff00000002, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48 }, // r+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^256-1
                { 0xc88b1a71b810d7b4, 0x5cf286851064fbc2, 0x993b9498361cc386, 0xc4d51fa950015f9f },
                { 0x60e55c7dd5623b5c, 0x33a1d44fb90ef709, 0xdc186fad1eaf837e, 0xfa211a6945f9fe12 },
                { 0x2a05f2556d58ede0, 0xd5347f13e6e27df5, 0x05276cfe630c9247, 0x524f0ac0e3bb7b9d },
                { 0xe2b5ffe82724e8f5, 0x47c1ec58ef0c159a, 0xdb1ccaa41fa78db2, 0x04c99214cc172e77 },
            },
            a[] = {
                { 0xfffffffe00000000, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2r-2
                { 0xfffffffe00000002, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2r
                { 0xfffffffe00000004, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2r+2
                { 0x00000003fffffffa, 0xb1096ff400069004, 0x33189fdfd9789fea, 0x304962b3598a0adf }, // 2^257-2
                { 0x911634e67021af65, 0xbeac210120cee388, 0x98c9a1184f53fefc, 0x2de14959232a4766 },
                { 0xc1cab8ffaac476b4, 0x184d189372247e16, 0xeb497f3a16d7a6e7, 0x248b9785e57e0704 },
                { 0x540be4abdab1dbbf, 0x56ab5a24cdc69feb, 0xd71501f4bc774c8a, 0x30b06e2e9dd979f1 },
                { 0xc56bffd04e49d1ea, 0x8f83d8b1de182b35, 0xb63995483f4f1b64, 0x09932429982e5cef },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<8); i++) {
            fr_cpy(t, q[i]);

            fr_x2(t);

            if (fr_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fr_x2: FAIL\n");
                printf("2 * 0x%016lx%016lx%016lx%016lx\n",
                q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx\n",
                a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx\n",
                t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fr_x3

    if (pass) {
        fr_t
            q[] = {
                { 0xffffffff00000000, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48 }, // r-1
                { 0xffffffff00000001, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48 }, // r
                { 0xffffffff00000002, 0x53bda402fffe5bfe, 0x3339d80809a1d805, 0x73eda753299d7d48 }, // r+1
                { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff }, // 2^256-1
                { 0x57dfda7aa31e3936, 0x09c6c5b91889a015, 0x78fc7321a936a1e5, 0xad831d7fec61991a },
                { 0xb77c61179fa1d259, 0x677a048e37480e6f, 0x3e432db3a97d45bd, 0x1d7c65d774b61a0f },
                { 0xc30ce006eb7c626f, 0xa9ec82098b214407, 0x19a27683d6e13e1c, 0xe2193f2301ead151 },
                { 0x336eda9dd2484c45, 0xda56edce88b8077a, 0x5696673b95c09810, 0x636811703bb2ce69 },
            },
            a[] = {
                { 0xfffffffdffffffff, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2p-3
                { 0xfffffffe00000002, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2p
                { 0xfffffffe00000005, 0xa77b4805fffcb7fd, 0x6673b0101343b00a, 0xe7db4ea6533afa90 }, // 2p+3
                { 0x00000005fffffff7, 0x098e27ee0009d806, 0xcca4efcfc634efe0, 0x486e140d064f104e }, // 3*2^256-3
                { 0x079f8f73e95aab9e, 0xce5dc11f49a37044, 0x9e0df944d51c8599, 0x38d2bb331eaed62e },
                { 0x26752346dee5770b, 0x366e0daaa5d82b4f, 0xbac9891afc77d138, 0x587531865e224e2d },
                { 0x4926a019c2752748, 0x5b11520da16c001c, 0x4cc62b63547a823b, 0x62a778c935ad018a },
                { 0x9a4c8fdb76d8e4cd, 0xe78981659a2b5e70, 0x9d4f85a2adfe1827, 0x425ce5aa5fdd70ab },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<8); i++) {
            fr_cpy(t, q[i]);

            fr_x3(t);

            if (fr_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fr_x3: FAIL\n");
                printf("3 * 0x%016lx%016lx%016lx%016lx\n",
                q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx\n",
                a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx\n",
                t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fr_add

    if (pass) {
        fr_t
            x[] = {
                { 0xa9adcd7739bd74cf, 0xb53be4eb9f0aba5b, 0xf39db86dd74ba55e, 0x205eeb7c38dbbf9d },
                { 0x1228a231f10f3969, 0x2f2fb5eebf0bbbe7, 0x0712c334b66a5db8, 0xebc3269a81507575 },
                { 0xb600a76ec90fb9ed, 0x07e79b33160b873e, 0x5a9d273d378355e4, 0x44583bcadb79bbb9 },
                { 0x504eb835fcb63c60, 0xbb06a2547d248706, 0xe9c922679104943d, 0xff4b832001a36d0d },
                { 0xe31f46556875472c, 0xaed37a617a390c6a, 0x3566711770d0337e, 0xcab450c115f31533 },
                { 0xccc5860e7b21e1c6, 0x58cadad8d1961158, 0x24fc2d78e8c1dd93, 0xc016a5c51c5a6199 },
                { 0x1f5ceaaf4f7d6de4, 0x4c2caf2b9b11fa95, 0xbff1464b3d3d6939, 0x795aa4384f5ab137 },
                { 0xf74a8f42e7f91926, 0x068815eb1cb349ca, 0xea0f7df125f638d4, 0x05dfc520280b525e },
            },
            y[] = {
                { 0xc4fec1d957afa21a, 0x620cdaf477088c3e, 0x81c14702fd352d79, 0x579c2cab14ba5662 },
                { 0x49031305fd08b422, 0x95247c6c292eafce, 0x66e5a460cc82bc8c, 0x8a6620ebfcebd51b },
                { 0xd7a40e39d1285b4b, 0xf2ea148ea21f8495, 0x33ccff2766b4f4c9, 0x719912e5a5aba394 },
                { 0xb0a9f4c1bd067f9e, 0xe57584bd59781e9a, 0x19ac2bbd2a9df826, 0x50a27f38bf3302ca },
                { 0x6ce53032663eb897, 0x00b44b033aff2e55, 0xa1bb974e40d1e8df, 0x0cab9a0b906628af },
                { 0x134aa1d8969bbf42, 0x8c36e257243dfcc3, 0x81cfbe199c2e992a, 0x797cd2e975ffcc1d },
                { 0x7db61267bd4d87f2, 0x364422505bd94d9e, 0x3196afb585723831, 0x900e218d7c6fa297 },
                { 0xa801fa7b481d5cca, 0x5e6809f0b1c9659f, 0x0087ad3b21681d47, 0xdf54afe9bc80f448 },
            },
            a[] = {
                { 0x6eac8f51916d16e8, 0xc38b1bdd1614ea9b, 0x42252768cadefad2, 0x040d70d423f898b8 },
                { 0x5b2bb53aee17ed88, 0xc91b4651e83f57b8, 0xd44adf7d66079234, 0x1a60518d0163d2b7 },
                { 0x8da4b5a99a381537, 0xa7140bbeb82cafd5, 0x5b304e5c949672a8, 0x4203a75d5787e205 },
                { 0x00f8acf9b9bcbbfc, 0xf900df0bd69feda3, 0x9d019e14a85edc59, 0x6812b3b26d9b7547 },
                { 0x50047688ceb3ffc2, 0x5bca2161b539dec1, 0xa3e8305da8004458, 0x637243797cbbc09a },
                { 0xe01027e911bda106, 0x3d867529f5d7561d, 0x40583b8271acc6b3, 0x51b82a083f1f3326 },
                { 0x9d12fd190ccaf5d4, 0xdaf58975f6ee9035, 0x8b1445f0af6bf15f, 0x218d771f788f593e },
                { 0x9f4c89bf301675ef, 0x11327bd8ce7e536b, 0xb75d53243dbc7e16, 0x7146cdb6baeec95e },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<8); i++) {
            fr_cpy(t, x[i]);

            fr_add(t, y[i]);

            if (fr_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fr_add: FAIL\n");
                printf("0x%016lx%016lx%016lx%016lx + 0x%016lx%016lx%016lx%016lx\n",
                x[i][3], x[i][2], x[i][1], x[i][0],
                y[i][3], y[i][2], y[i][1], y[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx\n",
                a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx\n",
                t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fr_sub

    if (pass) {
        fr_t
            x[] = {
                { 0x79c2dcd6c072a0eb, 0x5938348c63e5ee03, 0x2571278e41d8d960, 0x5cd48d8914135670 },
                { 0xdd0e7887bd1fae7d, 0x496ce21047d6852e, 0x6257ac6e44173664, 0x2c7ec7596a88e143 },
                { 0x4d1ae12876563e2f, 0x3544b4a08022241b, 0xdd5ac0fc26829b45, 0xde8c9d728a65a66b },
                { 0xe42f83194305299d, 0x2bf8fc83ec477308, 0x15c5d82b5e0d969b, 0x4549a8ae17749941 },
                { 0xcf43cbed7acb5065, 0xa62b1cef87f16d08, 0xce9aeb581a6f2561, 0x0aa57ffbe3f4eda8 },
                { 0xd76412152770edf7, 0xf28e9ae385c7b25e, 0x3ce8377fe91310d9, 0x76fb671eeccdab6f },
                { 0xe69ee87aff35023d, 0x1153b798996a5f4c, 0xde5d2875e048c920, 0x9932233eb0565fcd },
                { 0xa9d696814e42f2d7, 0x803049704c5e877a, 0xc4873604dbd43251, 0xe5096ba7b364ef06 },
            },
            y[] = {
                { 0x0525d4660db16d22, 0x5676c850dc0abe6a, 0x01d9e767cde7761d, 0x6adb7627e9f6aeef },
                { 0x00b82535d67c3d30, 0x4726bb54e06ddf10, 0x4e6c6c189328a37c, 0x500f5f5b67ab1495 },
                { 0x1a85df184987b062, 0x1f46f673e00873a1, 0x402323c56b47ff4d, 0xb56748bca526c301 },
                { 0xf2adc10711825b80, 0xa015fddb814bb5cc, 0x1170a2b515955e6e, 0x69ad05e1728edb01 },
                { 0x159ad40623ff267e, 0x9fee9451065ba006, 0xe7a0b5a0290501f7, 0x15225553b867d6ae },
                { 0x2706fb06c7aa6448, 0x02a555b27cc700f8, 0x2c3994f637e68acb, 0x7c27cf5b5ea73d45 },
                { 0x05aa619987706308, 0x9f3f454d588f659b, 0x17ed638bbc19695b, 0xccbf478a6c2b8bec },
                { 0x1e827c98a11f5038, 0xc8ea30df2b736be2, 0x280a417b3dcd44d1, 0x85f8ca724eb03970 },
            },
            a[] = {
                { 0x749d086fb2c133ca, 0x567f103e87d98b98, 0x56d1182e7d933b48, 0x65e6beb453ba24c9 },
                { 0xdc565350e6a3714e, 0x5603cabe6767021d, 0x4725185dba906aed, 0x505d0f512c7b49f6 },
                { 0x329502102cce8dcd, 0x15fdbe2ca019b07a, 0x9d379d36bb3a9bf8, 0x292554b5e53ee36a },
                { 0xf181c2113182ce1e, 0xdfa0a2ab6afa193a, 0x378f0d7e521a1031, 0x4f8a4a1fce833b88 },
                { 0xb9a8f7e656cc29e8, 0x59fa2ca181942901, 0x1a340dbffb0bfb6f, 0x6970d1fb552a9442 },
                { 0xb05d170d5fc689b0, 0x43a6e93408ff0d65, 0x43e87a91bace5e14, 0x6ec13f16b7c3eb72 },
                { 0xe0f486e077c49f36, 0xc5d2164e40d955b0, 0xf9a99cf22dd137c9, 0x406083076dc85129 },
                { 0x8b5419e8ad23a29f, 0xb746189120eb1b98, 0x9c7cf4899e06ed7f, 0x5f10a13564b4b596 },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<8); i++) {
            fr_cpy(t, x[i]);

            fr_sub(t, y[i]);

            if (fr_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fr_sub: FAIL\n");
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

    // fr_sqr

    if (pass) {
        fr_t
            q[] = {
                { 0x8a28bc7a23b8ca27, 0x2d5a18614d5d87dc, 0x0b5170efac5b0908, 0x5d142f76332013e5 },
                { 0x9e34a7d7bfd03f35, 0x3f8e1db0ff84a316, 0xc9041e4e1182ee8e, 0x24a6e8f76c1c85af },
                { 0x023f831fe9692b89, 0x8e0c0ea2e6efeccc, 0x4f7ef565a60c3557, 0x25407371bb909e31 },
                { 0x03f9bd4ad96916c6, 0xeeeb86b0e74a3539, 0x2877a95e76ce586d, 0x4d0458e170dbdd88 },
                { 0x7cd618ac00921af4, 0x2d5ef5cbd3296ec9, 0xc91d64d567eeeb7a, 0x4d30d5d0b742a756 },
                { 0x121e682e451a272c, 0x242abddbb631b4d1, 0x20e59d0cd3431333, 0xef33e5777473b021 },
                { 0xee35cf214032be18, 0x81248cd6767199a1, 0xaabeba2bc6b6700c, 0x16c0e36fc3e8a70a },
                { 0x2e0a1d0a32832d8e, 0xe7f1740cb372b6b8, 0xe92045e05b0b1ae1, 0x4f4034af157ec725 },
            },
            a[] = {
                { 0xead982e107652210, 0x46dd627d4957c30e, 0xa85d9d7f02a8b745, 0x4b38e9a5e46488a3 },
                { 0xc80b548070daf52c, 0x8eb33df3acf0542f, 0xb8b8b20de4b3d213, 0x034e5e23bf2e943f },
                { 0x8266bc13d1addffd, 0x77b3de68740601b0, 0x65db94345d920468, 0x46d10123c4906c55 },
                { 0x21a3dd420cff4523, 0x9d658b6044b8fb2a, 0x2e7e57b3eab308d1, 0x49a5d04eaea20b2e },
                { 0x946fd1562247103c, 0x820fa121380274bc, 0x209dccc4a056dd09, 0x5d162117e34178a8 },
                { 0x6c923642873be7d3, 0x39d5a34d22e476e3, 0xf9f58bacf3f1d0f3, 0x1f5926b571b05807 },
                { 0xdc3eb6dacd33e8fa, 0x105b0f90c65e9c36, 0x2b3a600e428534ce, 0x04eca66fa60bd82c },
                { 0x6c88a33e467d0436, 0x68de5bc32cb4b83d, 0x87e0d71ed13a8549, 0x0c6657a5ee1898ca },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<8); i++) {
            fr_cpy(t, q[i]);

            fr_sqr(t);

            if (fr_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fr_sqr: FAIL\n");
                printf("0x%016lx%016lx%016lx%016lx^2\n",
                q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx\n",
                a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx\n",
                t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fr_mul

    if (pass) {
        fr_t
            x[] = {
                { 0x3b44a96c59c690bb, 0x8697e87a732713a0, 0x945bea4e11648d71, 0x0c0aeeaacd9fdd46 },
                { 0x3b68c9ecd7f475dc, 0x8fcbf94cc47a7712, 0x98262bc012006f6b, 0xcc59b69c6a11800e },
                { 0x1c11a0b514009531, 0xbc79f3857c2d3e80, 0xb1b9ad274c9a1565, 0xabf34c7ef950991a },
                { 0x53b0394c29c2cced, 0xbd81e570393678b6, 0xc887cec9b33c9221, 0x9eb4079889d23a2b },
                { 0x161eb7e58a70f1be, 0x4d376cdcc50075ea, 0x421e12948915e8cb, 0xa94fe04dd3633195 },
                { 0x7796b5a9ce8ccc50, 0x89a5032329875fc0, 0xb37708d4e7c0695a, 0x55ad168e639da67c },
                { 0x5182e3055754826c, 0x244b84b1405dff5b, 0x7926637820bdf363, 0x0c4c38939953c602 },
                { 0x57c3d05787a089e4, 0x05923cbbec717106, 0xebb314273a513815, 0x68504f715af9afe4 },
                { 0x8775f85fefd3a3f7, 0x3d31cc4a7d359614, 0x0cbd12859051d3be, 0x18a286425c6a4f2a },
            },
            y[] = {
                { 0xa5acba7322aca411, 0x53166475e534bae2, 0x3c01f18e78a68d32, 0x2b26e44aea71c03d },
                { 0x390a4072d4249068, 0xed40e399d16d6bad, 0x6f45d1101c20d7c8, 0xf3892c2f5a40184b },
                { 0x772b7b7c9b803667, 0x8fb29be985c830d0, 0x40a28d850e1b986b, 0x0378847e6b358f44 },
                { 0x4170f9de04091161, 0x637f375b0fc38465, 0x3558e57591312120, 0xe007639449409917 },
                { 0xf25e79d416c742cf, 0x5c62c18fb9f86579, 0x0de4ea6a4912155d, 0xf8a3e25080ca559d },
                { 0xb8885f99f0bf3dee, 0x52ad47b58cc1fd03, 0xd645c5c990f5fcbd, 0xecfd9c6f82a56ce5 },
                { 0xa8580fe78a036375, 0xb0bf9f7b46b905c8, 0x990a16f11f2fdf81, 0x5a0a27df527959df },
                { 0x8bc3d4b107836984, 0xc274e53e00b19348, 0x86fcdbc0577d64e0, 0x534a17e34d281cd5 },
                { 0x0000000000000020, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
            },
            a[] = {
                { 0xde192398090105ca, 0xe59404bcd11fa709, 0xd92001ee4c393c0a, 0x139b5547381bff91 },
                { 0x2de6763e47381e72, 0xf94e924974a5b1df, 0xaa3357e41e101621, 0x3461aff32809ad73 },
                { 0x4b71a9acca4e8875, 0xcddb78d0cef1b76b, 0x498a2f1c6ab8c71d, 0x12be85e3b1f369e7 },
                { 0x6fea58bd903735bc, 0x44fe8b10a81a2cc2, 0xa5ab3007b5a7b6bd, 0x4dc3b2d38ae169d9 },
                { 0x3458bda5e25533f0, 0x8285e39ea2cdc36e, 0x145c2714ea99f555, 0x307c61d5ebb80b26 },
                { 0x83c19aee28b8e10b, 0xe8d1a598a5aa6d41, 0x2d9d4e82499c6082, 0x2f6125d87fec738a },
                { 0x1bc82ec8de6de6f6, 0x34994fa34fd514cc, 0xd106c6560453cd77, 0x24e2d06b5aafd115 },
                { 0xd35fd5461360911d, 0xedf0e88e620985e2, 0x7623693cae99b940, 0x03787c0416bdc9c6 },
                { 0xeebf0c03fa747eda, 0xafc7b13da6bc9a96, 0x64474081d06f67a7, 0x5cbedc589398f590 },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<9); i++) {
            fr_cpy(t, x[i]);

            fr_mul(t, y[i]);

            if (fr_neq(t, a[i]))
                pass = false;

            if (!pass) {
                printf("fr_mul: FAIL\n");
                printf("0x%016lx%016lx%016lx%016lx * 0x%016lx%016lx%016lx%016lx\n",
                x[i][3], x[i][2], x[i][1], x[i][0],
                y[i][3], y[i][2], y[i][1], y[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx\n",
                a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx\n",
                t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    // fr_inv

    if (pass) {
        fr_t
            q[] = {
                { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000002, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000003, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0xa127da24753d1ea9, 0xef5d7270e6940fb1, 0xc810db766a4a9105, 0x33344fab14e8ed3e },
                { 0x56131b9360c13570, 0x060d28a6bf60b67e, 0x9ad516f2aa1dbd56, 0x519d20d7a485985d },
                { 0x1a89fc6e8c94daed, 0x4db2fe4cec35a193, 0x46b68ee421699ad4, 0x664a986da9e91629 },
                { 0x7e26ae14d79bcb20, 0xca9b6bd2f2c0d033, 0x3cdef4136ccab1d9, 0x5039948ba7d60a02 },
            },
            a[] = {
                { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
                { 0x7fffffff80000001, 0xa9ded2017fff2dff, 0x199cec0404d0ec02, 0x39f6d3a994cebea4 },
                { 0xaaaaaaaa00000001, 0xe27e6d5755543d54, 0xccd13ab0066be558, 0x4d491a377113a8da },
                { 0x8cc00615238ce9a7, 0xe375dc0dbf0af565, 0xdba6b5f60714253a, 0x437bab96cfabdb5c },
                { 0x87e1447096dd8c82, 0x4ac035465169b630, 0xce62b1c33527e264, 0x5dd785ebc525afa2 },
                { 0x1db477d061fd382b, 0x4fed3127cdf422da, 0x4f85b52899197b2f, 0x41491bd8b882aa93 },
                { 0x6abb0a53dcaacf19, 0x8f9169fb559d59c1, 0xfc2bb52133daa733, 0x6476eac140621485 },
            };

        __shared__ fr_t t;

        for (int i=0; pass && (i<8); i++) {
            fr_cpy(t, q[i]);

            fr_inv(t);

            for (int j=0; j<4; j++)
                if (t[j] != a[i][j])
                    pass = false;

            if (!pass) {
                printf("fr_inv: FAIL\n");

                printf("1 / 0x%016lx%016lx%016lx%016lx\n",
                q[i][3], q[i][2], q[i][1], q[i][0]);

                printf("Expected 0x%016lx%016lx%016lx%016lx\n",
                a[i][3], a[i][2], a[i][1], a[i][0]);

                printf("Received 0x%016lx%016lx%016lx%016lx\n",
                t[3], t[2], t[1], t[0]);
            }

            ++count;
        }
    }

    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
