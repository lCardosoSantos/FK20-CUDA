// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fptest.cuh"

// a(b+c) = ab+ac

__global__ void FpTestAddDistributiveLeft(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;
    fp_t    a, b, c, u, v, w;

    for (int i=0; i<TESTVALS; i++) {
        fp_cpy(a, testval[i]);

        for (int j=0; j<TESTVALS; j++) {
            fp_cpy(b, testval[j]);

            for (int k=j; k<TESTVALS; k++) {
                fp_cpy(c, testval[k]);

                fp_cpy(u, a);
                fp_mul(u, u, b);   // ab

                fp_cpy(v, a);
                fp_mul(v, v, c);   // ac

                fp_add(u, u, v);   // ab+ac

                fp_cpy(v, a);
                fp_cpy(w, b);
                fp_add(w, w, c);   // b+c
                fp_mul(v, v, w);   // a(b+c)

                if (fp_neq(u, v)) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result\n", i, j);
                    printf("a = "); fp_print(testval[i]);
                    printf("b = "); fp_print(testval[j]);
                    printf("c = "); fp_print(testval[k]);
                    printf("ab+ac = "); fp_print(u);
                    printf("a(b+c) = "); fp_print(v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// (a+b)c = ac+bc

__global__ void FpTestAddDistributiveRight(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;
    fp_t    a, b, c, u, v;

    for (int i=0; i<TESTVALS; i++) {
        fp_cpy(a, testval[i]);

        for (int j=i; j<TESTVALS; j++) {
            fp_cpy(b, testval[j]);

            for (int k=0; k<TESTVALS; k++) {
                fp_cpy(c, testval[k]);

                fp_cpy(u, a);
                fp_mul(u, u, c);   // ac

                fp_cpy(v, b);
                fp_mul(v, v, c);   // bc

                fp_add(u, u, v);   // ac+bc

                fp_cpy(v, a);
                fp_add(v, v, b);   // a+b
                fp_mul(v, v, c);   // (a+b)c

                if (fp_neq(u, v)) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result\n", i, j);
                    printf("a = "); fp_print(testval[i]);
                    printf("b = "); fp_print(testval[j]);
                    printf("c = "); fp_print(testval[k]);
                    printf("ac+bc = "); fp_print(u);
                    printf("(a+b)c = "); fp_print(v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// a(b-c) = ab-ac

__global__ void FpTestSubDistributiveLeft(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;
    fp_t    a, b, c, u, v, w;

    for (int i=0; i<TESTVALS; i++) {
        fp_cpy(a, testval[i]);

        for (int j=0; j<TESTVALS; j++) {
            fp_cpy(b, testval[j]);

            for (int k=0; k<TESTVALS; k++) {
                fp_cpy(c, testval[k]);

                fp_cpy(u, a);
                fp_mul(u, u, b);   // ab

                fp_cpy(v, a);
                fp_mul(v, v, c);   // ac

                fp_sub(u, u, v);   // ab-ac

                fp_cpy(v, a);
                fp_cpy(w, b);
                fp_sub(w, w, c);   // b-c
                fp_mul(v, v, w);   // a(b-c)

                if (fp_neq(u, v)) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result\n", i, j);
                    printf("a = "); fp_print(testval[i]);
                    printf("b = "); fp_print(testval[j]);
                    printf("c = "); fp_print(testval[k]);
                    printf("ab-ac = "); fp_print(u);
                    printf("a(b-c) = "); fp_print(v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// (a-b)c = ac-bc

__global__ void FpTestSubDistributiveRight(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;
    fp_t    a, b, c, u, v;

    for (int i=0; i<TESTVALS; i++) {
        fp_cpy(a, testval[i]);

        for (int j=0; j<TESTVALS; j++) {
            fp_cpy(b, testval[j]);

            for (int k=0; k<TESTVALS; k++) {
                fp_cpy(c, testval[k]);

                fp_cpy(u, a);
                fp_mul(u, u, c);   // ac

                fp_cpy(v, b);
                fp_mul(v, v, c);   // bc

                fp_sub(u, u, v);   // ac-bc

                fp_cpy(v, a);
                fp_sub(v, v, b);   // a-b
                fp_mul(v, v, c);   // (a-b)c

                if (fp_neq(u, v)) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result\n", i, j);
                    printf("a = "); fp_print(testval[i]);
                    printf("b = "); fp_print(testval[j]);
                    printf("c = "); fp_print(testval[k]);
                    printf("ac-bc = "); fp_print(u);
                    printf("(a-b)c = "); fp_print(v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
