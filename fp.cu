// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fp.cuh"

/**
 * @brief Sets z to zero
 * 
 * @param[out] z 
 * @return void
 */
__device__ __host__ void fp_zero(fp_t &z) {
    for (int i=0; i<6; i++)
        z[i] = 0;
}

/**
 * @brief Sets z to one
 * 
 * @param[in,out] z 
 * @return __device__ 
 */
__device__ __host__ void fp_one(fp_t &z) {
    z[0] = 1;
    for (int i=1; i<6; i++)
        z[i] = 0;
}

/**
 * @brief Prints the canonical representation of x to STDOUT
 * 
 * @param[in] s Description string
 * @param[in] x Residue modulo p
 * @return void 
 */
__device__ void fp_print(const char *s, const fp_t &x) {
    fp_t t;
    fp_cpy(t, x);
    fp_reduce6(t);
    printf("%s", s);
//  printf("#x%016lx%016lx%016lx%016lx%016lx%016lx\n",  // clisp
    printf("%016lX%016lX%016lX%016lX%016lX%016lX\n",    // dc
//  printf("0x%016lx%016lx%016lx%016lx%016lx%016lx\n",  // python
    t[5], t[4], t[3], t[2], t[1], t[0]);
}

/**
 * @brief Converts from uint64_t[6] to a residue modulo p, without reduction.
 * 
 * @param[out] z fp_t residue modulo p
 * @param[in] x array of uint64_t 
 * @return __device__ 
 */
__device__ __host__ void fp_fromUint64(fp_t &z, const uint64_t *x) {
    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
    z[4] = x[4];
    z[5] = x[5];
}

/**
 * @brief Converts from residue modulo p (fp_t) to uint64_t[6]. The converted value
 * is in canonical form
 * 
 * @param[in] x 
 * @param[out] z 
 * @return void
 */
__device__ void fp_toUint64(uint64_t *z, const fp_t &x) {
    fp_t t;
    fp_cpy(t, x);
    fp_reduce6(t);

    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
    z[4] = x[4];
    z[5] = x[5];
}

// vim: ts=4 et sw=4 si
