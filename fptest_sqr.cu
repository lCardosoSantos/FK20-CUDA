// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

__global__ void FpTestSqr(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    const fp_t
        _1  = {  1, 0, 0, 0, 0, 0 },
        _2  = {  2, 0, 0, 0, 0, 0 },
        _4  = {  4, 0, 0, 0, 0, 0 },
        _6  = {  6, 0, 0, 0, 0, 0 },
        _16 = { 16, 0, 0, 0, 0, 0 },
        _36 = { 36, 0, 0, 0, 0, 0 };

    fp_t x, xsqr, x2, x4, x8, x12, l, r;

    // (x+n)^2 == x^2 + 2nx + n^2

    for (int i=0; pass && i<TESTVALS; i++) {

        fp_cpy(x, testval[i]);

        fp_sqr(xsqr, x);
        fp_x2(x2, x);   // n = 1
        fp_x4(x4, x);   // n = 2
        fp_x8(x8, x);   // n = 4
        fp_x12(x12, x); // n = 6

        // l = (x+1)^2
        fp_add(l, x, _1);
        fp_sqr(l, l);

        // r = x^2 + 2x + 1
        fp_add(r, xsqr, x2);
        fp_add(r, r, _1);

        if (fp_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            printf("x        : "); fp_print(x);
            printf("(x+1)^2  : "); fp_print(l);
            printf("x^2+2x+1 : "); fp_print(r);
            break;
        }
        ++count;

        // l = (x+2)^2
        fp_add(l, x, _2);
        fp_sqr(l, l);

        // r = x^2 + 4x + 4
        fp_add(r, xsqr, x4);
        fp_add(r, r, _4);

        if (fp_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            printf("x        : "); fp_print(x);
            printf("(x+2)^2  : "); fp_print(l);
            printf("x^2+4x+4 : "); fp_print(r);
            break;
        }
        ++count;

        // l = (x+4)^2
        fp_add(l, x, _4);
        fp_sqr(l, l);

        // r = x^2 + 8x + 16
        fp_add(r, xsqr, x8);
        fp_add(r, r, _16);

        if (fp_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            printf("x         : "); fp_print(x);
            printf("(x+4)^2   : "); fp_print(l);
            printf("x^2+8x+16 : "); fp_print(r);
            break;
        }
        ++count;

        // l = (x+6)^2
        fp_add(l, x, _6);
        fp_sqr(l, l);

        // r = x^2 + 12x + 36
        fp_add(r, xsqr, x12);
        fp_add(r, r, _36);

        if (fp_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            printf("x          : "); fp_print(x);
            printf("(x+6)^2    : "); fp_print(l);
            printf("x^2+12x+36 : "); fp_print(r);
            break;
        }
        ++count;
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

__global__ void FpTestSqr2(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fp_t x, xsqr, x2, y, l, r;

    // (x+y)^2 == x^2 + 2xy + y^2

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_cpy(x, testval[i]);
        fp_sqr(xsqr, x);
        fp_x2(x2, x);

        for (int j=i; pass && j<TESTVALS; j++) {

            // l = (x+y)^2
            fp_add(l, x, y);
            fp_sqr(l, l);

            // r = x^2 + 2xy + y^2
            fp_add(r, x2, y);   // 2x+y
            fp_mul(r, r, y);    // 2xy+y^2
            fp_add(r, xsqr, r);

            if (fp_neq(l, r)) {
                pass = false;

                printf("%d: FAILED\n", i);
                printf("x           : "); fp_print(x);
                printf("y           : "); fp_print(y);
                printf("(x+y)^2     : "); fp_print(l);
                printf("x^2+2xy+y^2 : "); fp_print(r);
                break;
            }
            ++count;
        }
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
