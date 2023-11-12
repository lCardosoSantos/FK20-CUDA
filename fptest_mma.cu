// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fptest.cuh"

/**
 * @brief Test for multiply-multiply-add. Compare with standalone
 * implementation of multiplication and addition functions.
 * 
 * @param testval 
 * @return void 
 */
__global__ void FpTestMMA(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=(TESTVALS > 48 ? TESTVALS-48 : 0); pass && i<TESTVALS; i++) {
        uint64_t v[6];

        fp_cpy(v, testval[i]);

        for (int j=i+1; pass && j<TESTVALS; j++) {
            uint64_t w[6], t[6];

            fp_cpy(w, testval[j]);
            fp_mul(t, v, w);

            for (int k=j+1; pass && k<TESTVALS; k++) {
                uint64_t x[6];

                fp_cpy(x, testval[k]);

                for (int l=k+1; pass && l<TESTVALS; l++) {
                    uint64_t y[6], u[6];

                    fp_cpy(y, testval[l]);
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

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
