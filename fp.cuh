// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FP_CUH
#define FP_CUH

#include <stdint.h>

typedef uint64_t fp_t[6];

extern __device__ void fp_fromUint64(uint64_t *z, const uint64_t *x);
extern __device__ void fp_toUint64(uint64_t *x, const uint64_t *z);
extern __device__ void fp_cpy(uint64_t *z, const uint64_t *x);
extern __device__ void fp_reduce6(uint64_t *z);
extern __device__ void fp_neg(uint64_t *z, const uint64_t *x);
extern __device__ void fp_x2(uint64_t *z, const uint64_t *x);
extern __device__ void fp_x3(uint64_t *z, const uint64_t *x);
extern __device__ void fp_x4(uint64_t *z, const uint64_t *x);
extern __device__ void fp_x8(uint64_t *z, const uint64_t *x);
extern __device__ void fp_x12(uint64_t *z, const uint64_t *x);
extern __device__ void fp_add(uint64_t *z, const uint64_t *x, const uint64_t *y);
extern __device__ void fp_sub(uint64_t *z, const uint64_t *x, const uint64_t *y);
extern __device__ void fp_sqr(uint64_t *z, const uint64_t *x);
extern __device__ void fp_mul(uint64_t *z, const uint64_t *x, const uint64_t *y);
extern __device__ void fp_mma(uint64_t *z, const uint64_t *v, const uint64_t *w, const uint64_t *x, const uint64_t *y);
extern __device__ void fp_inv(uint64_t *z, const uint64_t *x);
extern __device__ void fp_zero(uint64_t *z);
extern __device__ void fp_one(uint64_t *z);

extern __device__ bool fp_eq(uint64_t *x, uint64_t *y);
extern __device__ bool fp_neq(uint64_t *x, uint64_t *y);
extern __device__ bool fp_nonzero(const uint64_t *x);
extern __device__ bool fp_iszero(const uint64_t *x);
extern __device__ bool fp_isone(const uint64_t *x);

extern __device__ void fp_print(const uint64_t *x);

#endif
// vim: ts=4 et sw=4 si
