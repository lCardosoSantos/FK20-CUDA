// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

__global__ void FpTestCmp(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=500; i<770; i++) {
        uint64_t x[6];

        fp_cpy(x, testval[i].val);

        for (int j=500; j<770; j++) {
            uint64_t y[6];

            fp_cpy(y, testval[j].val);

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
                testval[i].val[5], testval[i].val[4], testval[i].val[3], testval[i].val[2], testval[i].val[1], testval[i].val[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                x[5], x[4], x[3], x[2], x[1], x[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                testval[j].val[5], testval[j].val[4], testval[j].val[3], testval[j].val[2], testval[j].val[1], testval[j].val[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                y[5], y[4], y[3], y[2], y[1], y[0]);

                printf("eq = %lx, neq = %lx\n", eq, neq);
            }

            if ((i != j) && eq) {
                pass = false;

                printf("%d,%d: FAIL B: fp_eq claims equality between these values:\n", i, j);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                testval[i].val[5], testval[i].val[4], testval[i].val[3], testval[i].val[2], testval[i].val[1], testval[i].val[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                x[5], x[4], x[3], x[2], x[1], x[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                testval[j].val[5], testval[j].val[4], testval[j].val[3], testval[j].val[2], testval[j].val[1], testval[j].val[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                y[5], y[4], y[3], y[2], y[1], y[0]);

                printf("eq = %lx, neq = %lx\n", eq, neq);
            }

            if ((i == j) && neq) {
                pass = false;

                printf("%d,%d: FAIL C: fp_neq claims inequality between these values:\n", i, j);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                testval[i].val[5], testval[i].val[4], testval[i].val[3], testval[i].val[2], testval[i].val[1], testval[i].val[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                x[5], x[4], x[3], x[2], x[1], x[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                testval[j].val[5], testval[j].val[4], testval[j].val[3], testval[j].val[2], testval[j].val[1], testval[j].val[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                y[5], y[4], y[3], y[2], y[1], y[0]);

                printf("eq = %lx, neq = %lx\n", eq, neq);
            }

            if ((i != j) && !neq) {
                pass = false;

                printf("%d,%d: FAIL D: fp_neq claims equality between these values:\n", i, j);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                testval[i].val[5], testval[i].val[4], testval[i].val[3], testval[i].val[2], testval[i].val[1], testval[i].val[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                x[5], x[4], x[3], x[2], x[1], x[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX/\n",
                testval[j].val[5], testval[j].val[4], testval[j].val[3], testval[j].val[2], testval[j].val[1], testval[j].val[0]);

                printf("\t%016lX%016lX%016lX%016lX%016lX%016lX\n",
                y[5], y[4], y[3], y[2], y[1], y[0]);

                printf("eq = %lx, neq = %lx\n", eq, neq);
            }
            ++count;
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
