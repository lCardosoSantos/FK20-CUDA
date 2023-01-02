// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"
#include "frtest.cuh"

__global__ void FrTestCmp(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;

    for (int i=2; i<514; i++) {
        fr_t x;

        fr_cpy(x, testval[i].val);

        for (int j=2; j<514; j++) {
            fr_t y;

            fr_cpy(y, testval[j].val);

            uint64_t
                eq  = fr_eq (x, y),
                neq = fr_neq(x, y);

            if (eq == neq) {
                pass = false;

                printf("%d,%d: FAILED: inconsistent result, eq = %lx, neq = %lx\n", i, j, eq, neq);
            }

            if ((i == j) && !eq) {
                pass = false;

                printf("%d,%d: FAIL A: fr_eq claims inequality between these values:\n", i, j);

                printf("\t%016lX%016lX%016lX%016lX/\n",
                testval[i].val[3], testval[i].val[2], testval[i].val[1], testval[i].val[0]);

                printf("\t%016lX%016lX%016lX%016lX\n",
                x[3], x[2], x[1], x[0]);

                printf("\t%016lX%016lX%016lX%016lX/\n",
                testval[j].val[3], testval[j].val[2], testval[j].val[1], testval[j].val[0]);

                printf("\t%016lX%016lX%016lX%016lX\n",
                y[3], y[2], y[1], y[0]);

                printf("eq = %lx, neq = %lx\n", eq, neq);
            }

            if ((i != j) && eq) {
                pass = false;

                printf("%d,%d: FAIL B: fr_eq claims equality between these values:\n", i, j);

                printf("\t%016lX%016lX%016lX%016lX/\n",
                testval[i].val[3], testval[i].val[2], testval[i].val[1], testval[i].val[0]);

                printf("\t%016lX%016lX%016lX%016lX\n",
                x[3], x[2], x[1], x[0]);

                printf("\t%016lX%016lX%016lX%016lX/\n",
                testval[j].val[3], testval[j].val[2], testval[j].val[1], testval[j].val[0]);

                printf("\t%016lX%016lX%016lX%016lX\n",
                y[3], y[2], y[1], y[0]);

                printf("eq = %lx, neq = %lx\n", eq, neq);
            }

            if ((i == j) && neq) {
                pass = false;

                printf("%d,%d: FAIL C: fr_neq claims inequality between these values:\n", i, j);

                printf("\t%016lX%016lX%016lX%016lX/\n",
                testval[i].val[3], testval[i].val[2], testval[i].val[1], testval[i].val[0]);

                printf("\t%016lX%016lX%016lX%016lX\n",
                x[3], x[2], x[1], x[0]);

                printf("\t%016lX%016lX%016lX%016lX/\n",
                testval[j].val[3], testval[j].val[2], testval[j].val[1], testval[j].val[0]);

                printf("\t%016lX%016lX%016lX%016lX\n",
                y[3], y[2], y[1], y[0]);

                printf("eq = %lx, neq = %lx\n", eq, neq);
            }

            if ((i != j) && !neq) {
                pass = false;

                printf("%d,%d: FAIL D: fr_neq claims equality between these values:\n", i, j);

                printf("\t%016lX%016lX%016lX%016lX/\n",
                testval[i].val[3], testval[i].val[2], testval[i].val[1], testval[i].val[0]);

                printf("\t%016lX%016lX%016lX%016lX\n",
                x[3], x[2], x[1], x[0]);

                printf("\t%016lX%016lX%016lX%016lX/\n",
                testval[j].val[3], testval[j].val[2], testval[j].val[1], testval[j].val[0]);

                printf("\t%016lX%016lX%016lX%016lX\n",
                y[3], y[2], y[1], y[0]);

                printf("eq = %lx, neq = %lx\n", eq, neq);
            }
            ++count;
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
