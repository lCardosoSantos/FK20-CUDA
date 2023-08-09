// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

__global__ void FpTestAdd(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fp_t x, l, r;

    // 2x + x == 3x

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_cpy(x, testval[i]);

        fp_x2(l, x);
        fp_add(l, l, x);

        fp_x3(r, x);

        if (fp_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            fp_print("x    : ",  x);
            fp_print("2x+x : ",  l);
            fp_print("3x   : ",  r);
        }
        ++count;
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// x+y == y+x

__global__ void FpTestCommutativeAdd(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=0; i<TESTVALS; i++) {
        for (int j=0; j<TESTVALS; j++) {
            fp_t x, y;

            fp_cpy(x, testval[i]);
            fp_cpy(y, testval[j]);

            fp_add(x, x, testval[j]);  // x + y
            fp_add(y, y, testval[i]);  // y + x

            if (fp_neq(x, y)) {
                pass = false;

                printf("%d,%d: FAILED: inconsistent result\n", i, j);
                fp_print("x = ",  testval[i]);
                fp_print("y = ",  testval[j]);
                fp_print("x+y = ",  x);
                fp_print("y+x = ",  y);
            }
            ++count;
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// (x+y)+z == x+(y+z)

__global__ void FpTestAssociativeAdd(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=0; i<TESTVALS; i++) {
        for (int j=0; j<TESTVALS; j++) {
            for (int k=0; k<TESTVALS; k++) {
                fp_t a, b, c;

                fp_cpy(a, testval[i]);  // x
                fp_cpy(b, testval[j]);  // y
                fp_cpy(c, testval[i]);  // x

                fp_add(a, a, testval[j]);  // x + y
                fp_add(a, a, testval[k]);  // (x + y) + z

                fp_add(b, b, testval[k]);  // y + z
                fp_add(c, c, b);           // x + (y + z)

                if (fp_neq(a, c)) {
                    pass = false;

                    printf("%d,%d,%d: FAILED: inconsistent result\n", i, j, k);
                    fp_print("x = ",  testval[i]);
                    fp_print("y = ",  testval[j]);
                    fp_print("z = ",  testval[k]);
                    fp_print("(x+y)+z = ",  a);
                    fp_print("x+(y+z) = ",  c);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
