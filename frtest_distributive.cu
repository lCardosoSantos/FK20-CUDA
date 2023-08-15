// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "frtest.cuh"


/**
 * @brief Check the distributive property (left of addition):
 * 
 * a(b+c) = ab+ac
 * 
 * @param testval 
 * @return void 
 */
__global__ void FrTestAddDistributiveLeft(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;
    fr_t    a, b, c, u, v, w;

    for (int i=0; i<TESTVALS; i++) {
        fr_cpy(a, testval[i]);

        for (int j=0; j<TESTVALS; j++) {
            fr_cpy(b, testval[j]);

            for (int k=j; k<TESTVALS; k++) {
                fr_cpy(c, testval[k]);

                fr_cpy(u, a);
                fr_mul(u, b);   // ab

                fr_cpy(v, a);
                fr_mul(v, c);   // ac

                fr_add(u, v);   // ab+ac

                fr_cpy(v, a);
                fr_cpy(w, b);
                fr_add(w, c);   // b+c
                fr_mul(v, w);   // a(b+c)

                if (fr_neq(u, v)) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result\n", i, j);
                    fr_print("a = ",  testval[i]);
                    fr_print("b = ",  testval[j]);
                    fr_print("c = ",  testval[k]);
                    fr_print("ab+ac = ",  u);
                    fr_print("a(b+c) = ",  v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    PRINTPASS(pass);
}

/**
 * @brief Check the distributive property (right of addition):
 * 
 * (a+b)c = ac+bc
 * 
 * @param testval 
 * @return void 
 */
__global__ void FrTestAddDistributiveRight(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;
    fr_t    a, b, c, u, v;

    for (int i=0; i<TESTVALS; i++) {
        fr_cpy(a, testval[i]);

        for (int j=i; j<TESTVALS; j++) {
            fr_cpy(b, testval[j]);

            for (int k=0; k<TESTVALS; k++) {
                fr_cpy(c, testval[k]);

                fr_cpy(u, a);
                fr_mul(u, c);   // ac

                fr_cpy(v, b);
                fr_mul(v, c);   // bc

                fr_add(u, v);   // ac+bc

                fr_cpy(v, a);
                fr_add(v, b);   // a+b
                fr_mul(v, c);   // (a+b)c

                if (fr_neq(u, v)) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result\n", i, j);
                    fr_print("a = ",  testval[i]);
                    fr_print("b = ",  testval[j]);
                    fr_print("c = ",  testval[k]);
                    fr_print("ac+bc = ",  u);
                    fr_print("(a+b)c = ",  v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}


/**
 * @brief Check the distributive property (left of subtraction):
 * 
 * a(b-c) = ab-ac
 * 
 * @param testval 
 * @return void 
 */
__global__ void FrTestSubDistributiveLeft(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;
    fr_t    a, b, c, u, v, w;

    for (int i=0; i<TESTVALS; i++) {
        fr_cpy(a, testval[i]);

        for (int j=0; j<TESTVALS; j++) {
            fr_cpy(b, testval[j]);

            for (int k=0; k<TESTVALS; k++) {
                fr_cpy(c, testval[k]);

                fr_cpy(u, a);
                fr_mul(u, b);   // ab

                fr_cpy(v, a);
                fr_mul(v, c);   // ac

                fr_sub(u, v);   // ab-ac

                fr_cpy(v, a);
                fr_cpy(w, b);
                fr_sub(w, c);   // b-c
                fr_mul(v, w);   // a(b-c)

                if (fr_neq(u, v)) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result\n", i, j);
                    fr_print("a = ",  testval[i]);
                    fr_print("b = ",  testval[j]);
                    fr_print("c = ",  testval[k]);
                    fr_print("ab-ac = ",  u);
                    fr_print("a(b-c) = ",  v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// (a-b)c = ac-bc

/**
 * @brief Check the distributive property (right of subtraction):
 * 
 * (a-b)c = ac-bc
 * 
 * @param testval 
 * @return void 
 */
__global__ void FrTestSubDistributiveRight(testval_t *testval) {

    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;
    fr_t    a, b, c, u, v;

    for (int i=0; i<TESTVALS; i++) {
        fr_cpy(a, testval[i]);

        for (int j=0; j<TESTVALS; j++) {
            fr_cpy(b, testval[j]);

            for (int k=0; k<TESTVALS; k++) {
                fr_cpy(c, testval[k]);

                fr_cpy(u, a);
                fr_mul(u, c);   // ac

                fr_cpy(v, b);
                fr_mul(v, c);   // bc

                fr_sub(u, v);   // ac-bc

                fr_cpy(v, a);
                fr_sub(v, b);   // a-b
                fr_mul(v, c);   // (a-b)c

                if (fr_neq(u, v)) {
                    pass = false;

                    printf("%d,%d: FAILED: inconsistent result\n", i, j);
                    fr_print("a = ",  testval[i]);
                    fr_print("b = ",  testval[j]);
                    fr_print("c = ",  testval[k]);
                    fr_print("ac-bc = ",  u);
                    fr_print("(a-b)c = ",  v);
                }
                ++count;
            }
        }
    }
    printf("%ld tests\n", count);

    printf("--- %s: %s\n", pass ? "PASS" : "FAIL", __func__);
}

// vim: ts=4 et sw=4 si
