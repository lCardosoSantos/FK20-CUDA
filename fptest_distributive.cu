// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fptest.cuh"
#include "fp_ptx.cuh"

/**
 * @brief Check the distributive property of multiplication in Fp (left of addition):
 * 
 * a(b+c) = ab+ac
 * 
 * @param testval 
 * @return void 
 */
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

/**
 * @brief Check the distributive property of multiplication in Fp (right of addition):
 * 
 * (a+b)c = ac+bc
 * 
 * @param testval 
 * @return void 
 */
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

/**
 * @brief Check the distributive property of multiplication in Fp (left of subtraction):
 * 
 * a(b-c) = ab-ac
 * 
 * @param testval 
 * @return void 
 */
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

/**
 * @brief Check the distributive property of multiplication in Fp (right of subtraction):
 * 
 * (a-b)c = ac-bc
 * 
 * @param testval 
 * @return void 
 */
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


/**
 * @brief Check the distributive property of multiplication in Fp using PTX macros (left of addition):
 * 
 * a(b+c) = ab+ac
 * 
 * @param testval 
 * @return void 
 */
__global__ void FpTestAddDistributiveLeftPTX(testval_t *testval) {

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
                fp_mul_ptx(u, u, b);   // ab

                fp_cpy(v, a);
                fp_mul_ptx(v, v, c);   // ac

                fp_add_ptx(u, u, v);   // ab+ac

                fp_cpy(v, a);
                fp_cpy(w, b);
                fp_add_ptx(w, w, c);   // b+c
                fp_mul_ptx(v, v, w);   // a(b+c)

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

/**
 * @brief Check the distributive property of multiplication in Fp using PTX macros (right of addition):
 * 
 * (a+b)c = ac+bc
 * 
 * @param testval 
 * @return void 
 */
__global__ void FpTestAddDistributiveRightPTX(testval_t *testval) {

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
                fp_mul_ptx(u, u, c);   // ac

                fp_cpy(v, b);
                fp_mul_ptx(v, v, c);   // bc

                fp_add_ptx(u, u, v);   // ac+bc

                fp_cpy(v, a);
                fp_add_ptx(v, v, b);   // a+b
                fp_mul_ptx(v, v, c);   // (a+b)c

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

/**
 * @brief Check the distributive property of multiplication in Fp using PTX macros (left of subtraction):
 * 
 * a(b-c) = ab-ac
 * 
 * @param testval 
 * @return void 
 */
__global__ void FpTestSubDistributiveLeftPTX(testval_t *testval) {

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
                fp_mul_ptx(u, u, b);   // ab

                fp_cpy(v, a);
                fp_mul_ptx(v, v, c);   // ac

                fp_sub(u, u, v);   // ab-ac

                fp_cpy(v, a);
                fp_cpy(w, b);
                fp_sub(w, w, c);   // b-c
                fp_mul_ptx(v, v, w);   // a(b-c)

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

/**
 * @brief Check the distributive property of multiplication in Fp using PTX macros `(right of subtraction):
 * 
 * (a-b)c = ac-bc
 * 
 * @param testval 
 * @return void 
 */
__global__ void FpTestSubDistributiveRightPTX(testval_t *testval) {

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
                fp_mul_ptx(u, u, c);   // ac

                fp_cpy(v, b);
                fp_mul_ptx(v, v, c);   // bc

                fp_sub(u, u, v);   // ac-bc

                fp_cpy(v, a);
                fp_sub(v, v, b);   // a-b
                fp_mul_ptx(v, v, c);   // (a-b)c

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
