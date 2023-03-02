// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FR_CUH
#define FR_CUH

#include <stdint.h>

typedef uint64_t fr_t[4];

extern __constant__ fr_t fr_roots[515];

extern __device__ void fr_fromUint64(fr_t &z, const uint64_t *x);
extern __device__ void fr_toUint64(const fr_t &x, uint64_t *z);
extern __device__ void fr_cpy(fr_t &z, const fr_t &x);
extern __device__ void fr_reduce4(fr_t &z);
extern __device__ void fr_neg(fr_t &z);
extern __device__ void fr_x2(fr_t &z);
extern __device__ void fr_x3(fr_t &z);
extern __device__ void fr_x4(fr_t &z);
extern __device__ void fr_x8(fr_t &z);
extern __device__ void fr_x12(fr_t &z);
extern __device__ void fr_add(fr_t &z, const fr_t &x);
extern __device__ void fr_sub(fr_t &z, const fr_t &x);
extern __device__ void fr_addsub(fr_t &x, fr_t &y);
extern __device__ void fr_sqr(fr_t &z);
extern __device__ void fr_mul(fr_t &z, const fr_t &x);
extern __device__ void fr_inv(fr_t &z);
extern __device__ void fr_zero(fr_t &z);
extern __device__ void fr_one(fr_t &z);

extern __device__ bool fr_eq(const fr_t &x, const fr_t &y);
extern __device__ bool fr_neq(const fr_t &x, const fr_t &y);
extern __device__ bool fr_nonzero(const fr_t &x);
extern __device__ bool fr_iszero(const fr_t &x);
extern __device__ bool fr_isone(const fr_t &x);

extern __device__ void fr_print(const fr_t &x);

#endif

// vim: ts=4 et sw=4 si
