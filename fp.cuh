// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_CUH
#define FP_CUH

#include <stdint.h>

/**
 * @brief Residue modulo p. Any 384-bit representative of each residue is
 * allowed, and stored as a 6-element little-endian array of uint64_t.
 */
typedef uint64_t fp_t[6];

extern __device__ __host__ void fp_fromUint64(fp_t &z, const uint64_t *x);
extern __device__ void fp_toUint64(uint64_t *z, const fp_t &x);
extern __device__ __host__ void fp_cpy(fp_t &z, const fp_t &x);
extern __device__ void fp_reduce6(fp_t &z);
extern __device__ void fp_neg(fp_t &z, const fp_t &x);
extern __device__ void fp_x2(fp_t &z, const fp_t &x);
extern __device__ void fp_x3(fp_t &z, const fp_t &x);
extern __device__ void fp_x4(fp_t &z, const fp_t &x);
extern __device__ void fp_x8(fp_t &z, const fp_t &x);
extern __device__ void fp_x12(fp_t &z, const fp_t &x);
extern __device__ void fp_add(fp_t &z, const fp_t &x, const fp_t &y);
extern __device__ void fp_sub(fp_t &z, const fp_t &x, const fp_t &y);
extern __device__ void fp_sqr(fp_t &z, const fp_t &x);
extern __device__ void fp_mul(fp_t &z, const fp_t &x, const fp_t &y);
extern __device__ void fp_mma(fp_t &z, const fp_t &v, const fp_t &w, const fp_t &x, const fp_t &y);
extern __device__ void fp_inv(fp_t &z, const fp_t &x);
extern __device__ __host__ void fp_zero(fp_t &z);
extern __device__ __host__ void fp_one(fp_t &z);

extern __device__ bool fp_eq(const fp_t &x, const fp_t &y);
extern __device__ bool fp_neq(const fp_t &x, const fp_t &y);
extern __device__ bool fp_nonzero(const fp_t &x);
extern __device__ bool fp_iszero(const fp_t &x);
extern __device__ bool fp_isone(const fp_t &x);

extern __device__ void fp_print(const char *s, const fp_t &x);

#endif
// vim: ts=4 et sw=4 si
