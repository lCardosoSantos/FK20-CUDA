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
                    fp_print("a = ",  testval[i]);
                    fp_print("b = ",  testval[j]);
                    fp_print("c = ",  testval[k]);
                    fp_print("ab+ac = ",  u);
                    fp_print("a(b+c) = ",  v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
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
                    fp_print("a = ",  testval[i]);
                    fp_print("b = ",  testval[j]);
                    fp_print("c = ",  testval[k]);
                    fp_print("ac+bc = ",  u);
                    fp_print("(a+b)c = ",  v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
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
                    fp_print("a = ",  testval[i]);
                    fp_print("b = ",  testval[j]);
                    fp_print("c = ",  testval[k]);
                    fp_print("ab-ac = ",  u);
                    fp_print("a(b-c) = ",  v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
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
                    fp_print("a = ",  testval[i]);
                    fp_print("b = ",  testval[j]);
                    fp_print("c = ",  testval[k]);
                    fp_print("ac-bc = ",  u);
                    fp_print("(a-b)c = ",  v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

// vim: ts=4 et sw=4 si
