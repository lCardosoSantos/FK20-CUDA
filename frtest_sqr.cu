// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "frtest.cuh"

/**
 * @brief Test for squaring on Fr. Checks for self consistency:
 *
 * (x+n)^2 == x^2 + 2nx + n^2
 *
 * @param testval
 * @return void
 */
__global__ void FrTestSqr(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    const fr_t
        _1  = {  1, 0, 0, 0 },
        _2  = {  2, 0, 0, 0 },
        _4  = {  4, 0, 0, 0 },
        _6  = {  6, 0, 0, 0 },
        _16 = { 16, 0, 0, 0 },
        _36 = { 36, 0, 0, 0 };

    fr_t x, xsqr, x2, x4, x8, x12, l, r;

    // (x+n)^2 == x^2 + 2nx + n^2

    for (int i=0; pass && i<TESTVALS; i++) {

        fr_cpy(x, testval[i]);
        fr_cpy(xsqr, x);
        fr_cpy(x2, x);
        fr_cpy(x4, x);
        fr_cpy(x8, x);
        fr_cpy(x12, x);

        fr_sqr(xsqr);
        fr_x2(x2);   // n = 1
        fr_x4(x4);   // n = 2
        fr_x8(x8);   // n = 4
        fr_x12(x12); // n = 6

        // l = (x+1)^2
        fr_cpy(l, x);
        fr_add(l, _1);
        fr_sqr(l);

        // r = x^2 + 2x + 1
        fr_cpy(r, xsqr);
        fr_add(r, x2);
        fr_add(r, _1);

        if (fr_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            fr_print("x        : ",  x);
            fr_print("(x+1)^2  : ",  l);
            fr_print("x^2+2x+1 : ",  r);
            break;
        }
        ++count;

        // l = (x+2)^2
        fr_cpy(l, x);
        fr_add(l, _2);
        fr_sqr(l);

        // r = x^2 + 4x + 4
        fr_cpy(r, xsqr);
        fr_add(r, x4);
        fr_add(r, _4);

        if (fr_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            fr_print("x        : ",  x);
            fr_print("(x+2)^2  : ",  l);
            fr_print("x^2+4x+4 : ",  r);
            break;
        }
        ++count;

        // l = (x+4)^2
        fr_cpy(l, x);
        fr_add(l, _4);
        fr_sqr(l);

        // r = x^2 + 8x + 16
        fr_cpy(r, xsqr);
        fr_add(r, x8);
        fr_add(r, _16);

        if (fr_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            fr_print("x         : ",  x);
            fr_print("(x+4)^2   : ",  l);
            fr_print("x^2+8x+16 : ",  r);
            break;
        }
        ++count;

        // l = (x+6)^2
        fr_cpy(l, x);
        fr_add(l, _6);
        fr_sqr(l);

        // r = x^2 + 12x + 36
        fr_cpy(r, xsqr);
        fr_add(r, x12);
        fr_add(r, _36);

        if (fr_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            fr_print("x          : ",  x);
            fr_print("(x+6)^2    : ",  l);
            fr_print("x^2+12x+36 : ",  r);
            break;
        }
        ++count;
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

/**
 * @brief Test for squaring on Fr. Checks for self consistency:
 *
 * (x+y)^2 == x^2 + 2xy + y^2
 *
 * @param testval
 * @return void
 */
__global__ void FrTestSqr2(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fr_t x, xsqr, x2, y, l, r;

    // (x+y)^2 == x^2 + 2xy + y^2

    for (int i=0; pass && i<TESTVALS; i++) {
        fr_cpy(x, testval[i]);
        fr_cpy(xsqr, x);
        fr_cpy(x2, x);

        fr_sqr(xsqr);
        fr_x2(x2);

        for (int j=i; pass && j<TESTVALS; j++) {

            // l = (x+y)^2
            fr_cpy(l, x);
            fr_add(l, y);
            fr_sqr(l);

            // r = x^2 + 2xy + y^2
            fr_cpy(r, x2);
            fr_add(r, y);   // 2x+y
            fr_mul(r, y);    // 2xy+y^2
            fr_add(r, xsqr);

            if (fr_neq(l, r)) {
                pass = false;

                printf("%d: FAILED\n", i);
                fr_print("x           : ",  x);
                fr_print("y           : ",  y);
                fr_print("(x+y)^2     : ",  l);
                fr_print("x^2+2xy+y^2 : ",  r);
                break;
            }
            ++count;
        }
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
