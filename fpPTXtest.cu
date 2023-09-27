// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos


// Tests if the inline PTX and pure PTX functions generate matching results.

#include "fp.cuh"
#include "fptest.cuh"
#include "fp_ptx.cuh"

/**
 * @brief Test the equivalence between the inline PTX functions with the pure PTX
 * macros
 * 
 * @return __global__ 
 */
__global__ void FpTestEqPTXInline(testval_t *testval) {
    printf("=== RUN   %s\n", __func__);

    bool    pass    = true;
    size_t  count   = 0;
    fp_t x, y, r, rp;
    //add
    for(int i=0; pass && i<TESTVALS; i++){
        for(int j=0; pass && j<TESTVALS; j++){
            fp_cpy(x, testval[i]);
            fp_cpy(y, testval[j]);

            fp_add(r, x, y);
            fp_add_ptx(rp, x, y);

            if(fp_neq(r, rp)){
                pass = false;

                printf("%d: FAILED add\n", i);
                fp_print("x    : ",  x);
                fp_print("y    : ",  y);
                fp_print("r    : ",  r);
                fp_print("rp   : ",  rp);
            }
            ++count;
        }
    }
    //sub
    for(int i=0; pass && i<TESTVALS; i++){
        for(int j=0; pass && j<TESTVALS; j++){
            fp_cpy(x, testval[i]);
            fp_cpy(y, testval[j]);

            fp_sub(r, x, y);
            fp_sub_ptx(rp, x, y);

            if(fp_neq(r, rp)){
                pass = false;

                printf("%d: FAILED sub\n", i);
                fp_print("x    : ",  x);
                fp_print("y    : ",  y);
                fp_print("r    : ",  r);
                fp_print("rp   : ",  rp);
            }
            ++count;
        }
    }
    //mul
    for(int i=0; pass && i<TESTVALS; i++){
        for(int j=0; pass && j<TESTVALS; j++){
            fp_cpy(x, testval[i]);
            fp_cpy(y, testval[j]);

            fp_mul(r, x, y);
            fp_mul_ptx(rp, x, y);

            if(fp_neq(r, rp)){
                pass = false;

                printf("%d: FAILED mul\n", i);
                fp_print("x    : ",  x);
                fp_print("y    : ",  y);
                fp_print("r    : ",  r);
                fp_print("rp   : ",  rp);
            }
            ++count;
        }
    }

    //sqr
    for(int i=0; pass && i<TESTVALS; i++){
            fp_cpy(x, testval[i]);

            fp_sqr(r, x);
            fp_sqr_ptx(rp, x);

            if(fp_neq(r, rp)){
                pass = false;

                printf("%d: FAILED sqr\n", i);
                fp_print("x    : ",  x);
                fp_print("y    : ",  y);
                fp_print("r    : ",  r);
                fp_print("rp   : ",  rp);
            }
            ++count;
    }

    //x2
    for(int i=0; pass && i<TESTVALS; i++){
            fp_cpy(x, testval[i]);

            fp_x2(r, x);
            fp_x2_ptx(rp, x);

            if(fp_neq(r, rp)){
                pass = false;

                printf("%d: FAILED x2\n", i);
                fp_print("x    : ",  x);
                fp_print("y    : ",  y);
                fp_print("r    : ",  r);
                fp_print("rp   : ",  rp);
            }
            ++count;
    }

    //x3
    for(int i=0; pass && i<TESTVALS; i++){
            fp_cpy(x, testval[i]);

            fp_x3(r, x);
            fp_x3_ptx(rp, x);

            if(fp_neq(r, rp)){
                pass = false;

                printf("%d: FAILED x2\n", i);
                fp_print("x    : ",  x);
                fp_print("y    : ",  y);
                fp_print("r    : ",  r);
                fp_print("rp   : ",  rp);
            }
            ++count;
    }

    //x4
    for(int i=0; pass && i<TESTVALS; i++){
            fp_cpy(x, testval[i]);

            fp_x4(r, x);
            fp_x4_ptx(rp, x);

            if(fp_neq(r, rp)){
                pass = false;

                printf("%d: FAILED x2\n", i);
                fp_print("x    : ",  x);
                fp_print("y    : ",  y);
                fp_print("r    : ",  r);
                fp_print("rp   : ",  rp);
            }
            ++count;
    }

    //x8
    for(int i=0; pass && i<TESTVALS; i++){
            fp_cpy(x, testval[i]);

            fp_x8(r, x);
            fp_x8_ptx(rp, x);

            if(fp_neq(r, rp)){
                pass = false;

                printf("%d: FAILED x2\n", i);
                fp_print("x    : ",  x);
                fp_print("y    : ",  y);
                fp_print("r    : ",  r);
                fp_print("rp   : ",  rp);
            }
            ++count;
    }

    //x12
    for(int i=0; pass && i<TESTVALS; i++){
            fp_cpy(x, testval[i]);

            fp_x12(r, x);
            fp_x12_ptx(rp, x);

            if(fp_neq(r, rp)){
                pass = false;

                printf("%d: FAILED x2\n", i);
                fp_print("x    : ",  x);
                fp_print("y    : ",  y);
                fp_print("r    : ",  r);
                fp_print("rp   : ",  rp);
            }
            ++count;
    }

    printf("%ld tests\n", count);

    PRINTPASS(pass);
}
