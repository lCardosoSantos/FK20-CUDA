// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fr.cuh"

/**
 * @brief Sets the value of z to zero
 * 
 * @param[out] z 
 * @return void 
 */
__device__ __host__ void fr_zero(fr_t &z) {
    for (int i=0; i<4; i++)
        z[i] = 0;
}

/**
 * @brief Sets the value of z to one.
 * 
 * @param[out] z 
 * @return void
 */
__device__ __host__ void fr_one(fr_t &z) {
    z[0] = 1;
    for (int i=1; i<4; i++)
        z[i] = 0;
}

/**
 * @brief prints the canonical representation of x to STDOUT
 * 
 * Prints the canonical hexadecimal representation of x to stdout, followed by linefeed; 
 * prints with leading zeros, and without the hex prefix.
 * @param[in] s Description string
 * @param[in] x 
 * @return void 
 */
__device__ void fr_print(const char *s, const fr_t &x) {
    fr_t t;
    fr_cpy(t, x);
    fr_reduce4(t);
    printf("%s", s);
    printf("%016lX%016lX%016lX%016lX\n",    // dc
//  printf("#x%016lx%016lx%016lx%016lx\n",  // clisp compatible format
//  printf("0x%016lx%016lx%016lx%016lx\n",  // python compatible format
    t[3], t[2], t[1], t[0]);
}

/**
 * @brief Converts uint64_t[4] to fr_t
 * 
 * Converts uint64_t[4] to fr_t. Word order is BIG endian.
 * 
 * @param[out] z Destination
 * @param[in] x Pointer to array to be converted.
 * @return void 
 */
__device__ __host__ void fr_fromUint64(fr_t &z, const uint64_t *x) {
    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
}

/**
 * @brief Converts fr_t to uint64_t[4]
 * 
 * Converts uint64_t[4] to fr_t. Word order is BIG endian. The 256-bit bit value
 * x is reduced before convert, the original value is unchanged.
 * 
 * @param[out] z Pointer to destination array
 * @param[in] x fr_t to be converted.
 * @return void 
 */
__device__ void fr_toUint64(const fr_t &x, uint64_t *z) {
    fr_t t;
    fr_cpy(t, x);
    fr_reduce4(t);

    z[0] = x[0];
    z[1] = x[1];
    z[2] = x[2];
    z[3] = x[3];
}

/**
 * @brief Checks equality of two arrays of fr_t, element wise, and store in a byte array
 * 
 * Uses the CUDA device to perform a fast comparision between two arrays of fr_t.
 * This function has no limitation on the number and size of blocks.
 * 
 * @param[out] eq Array of count bytes, such that eq[i]==1 if x[i] == y[i], zero otherwise.
 * @param[in] count Number of elements to be compared
 * @param[in] x First array
 * @param[in] y Second array
 * @return void 
 */
__global__ void fr_eq_wrapper(uint8_t *eq, int count, const fr_t *x, const fr_t *y) {

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    unsigned step = gridDim.z * gridDim.y * gridDim.x
                * blockDim.z * blockDim.y * blockDim.x;

    for (unsigned i=tid; i<count; i+=step)
        eq[i] = fr_eq(x[i], y[i]);
}

// vim: ts=4 et sw=4 si
