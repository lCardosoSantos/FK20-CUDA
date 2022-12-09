// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FR_CUH
#define FR_CUH

#include <stdint.h>

typedef uint64_t fr_t[4];

extern __device__ void fr_fromUint64(uint64_t *z, const uint64_t *x);
extern __device__ void fr_toUint64(uint64_t *x, const uint64_t *z);
extern __device__ void fr_cpy(uint64_t *z, const uint64_t *x);
extern __device__ void fr_reduce4(uint64_t *z);
extern __device__ void fr_neg(uint64_t *z);
extern __device__ void fr_x2(uint64_t *z);
extern __device__ void fr_x3(uint64_t *z);
extern __device__ void fr_x4(uint64_t *z);
extern __device__ void fr_x8(uint64_t *z);
extern __device__ void fr_x12(uint64_t *z);
extern __device__ void fr_add(uint64_t *z, const uint64_t *x);
extern __device__ void fr_sub(uint64_t *z, const uint64_t *x);
extern __device__ void fr_sqr(uint64_t *z);
extern __device__ void fr_mul(uint64_t *z, const uint64_t *x);
extern __device__ void fr_inv(uint64_t *z);
extern __device__ void fr_zero(uint64_t *z);
extern __device__ void fr_one(uint64_t *z);

extern __device__ bool fr_eq(uint64_t *x, uint64_t *y);
extern __device__ bool fr_neq(uint64_t *x, uint64_t *y);
extern __device__ bool fr_nonzero(const uint64_t *x);
extern __device__ bool fr_iszero(const uint64_t *x);
extern __device__ bool fr_isone(const uint64_t *x);

extern __device__ void fr_print(const uint64_t *x);

#endif

// vim: ts=4 et sw=4 si
