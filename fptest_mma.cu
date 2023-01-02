// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

__global__ void FpTestMMA(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=880; i<TESTVALS; i++) {
        uint64_t v[6];

        fp_cpy(v, testval[i].val);

        for (int j=i+1; j<TESTVALS; j++) {
            uint64_t w[6], t[6];

            fp_cpy(w, testval[j].val);
            fp_mul(t, v, w);

            for (int k=j+1; k<TESTVALS; k++) {
                uint64_t x[6];

                fp_cpy(x, testval[k].val);

                for (int l=k+1; l<TESTVALS; l++) {
                    uint64_t y[6], u[6];

                    fp_cpy(y, testval[l].val);
                    fp_mul(u, x, y);
                    fp_add(u, u, t);

                    fp_mma(y, v, w, x, y);

                    if (fp_neq(u, y)) {
                        pass = false;

                        printf("(%d,%d,%d,%d): FAILED\n", i, j, k, l);
                    }

                    ++count;
                }
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
