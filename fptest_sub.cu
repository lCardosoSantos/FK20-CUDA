// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

__global__ void FpTestSub(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    fp_t x, l, r;

    // 2x == 3x - x

    for (int i=0; pass && i<TESTVALS; i++) {
        fp_cpy(x, testval[i]);

        fp_x2(l, x);

        fp_x3(r, x);
        fp_sub(r, r, x);

        if (fp_neq(l, r)) {
            pass = false;

            printf("%d: FAILED\n", i);
            printf("x    : "); fp_print(x);
            printf("2x   : "); fp_print(l);
            printf("3x-x : "); fp_print(r);
        }
        ++count;
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
