// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef G1_CUH
#define G1_CUH

#include <stdint.h>

#include "fp.cuh"
#include "fr.cuh"

// Set this nonzero to allow any point at infinity, i.e. of the form (0, !0, 0)
#define G1P_ANYINF 1

/**
 * @brief G1 point in affine coordinates
 * 
 */
typedef struct {
    fp_t x, y;
} g1a_t;
/**
 * @brief G1 point in projective coordinates
 * 
 */
typedef struct {
    fp_t x, y, z;
} g1p_t;

extern __device__ __host__ void g1a_fromUint64(g1a_t &a, const uint64_t *x, const uint64_t *y);
extern __device__ __host__ void g1a_fromFp(g1a_t &a, const fp_t &x, const fp_t &y);
extern __device__ void g1a_fromG1p(g1a_t &a, const g1p_t &p);
extern __device__ __host__ void g1a_print(const char *s, const g1a_t &a);
extern __device__ __host__ void g1a_cpy(g1a_t &a, const g1a_t &b);

extern __device__ void g1p_toUint64(const g1p_t &p, uint64_t *x, uint64_t *y, uint64_t *z);
extern __device__ __host__ void g1p_fromUint64(g1p_t &p, const uint64_t *x, const uint64_t *y, const uint64_t *z);
inline __device__ __host__ void g1p_fromFp(g1p_t &p, fp_t &x, fp_t &y, fp_t &z) {
    g1p_fromUint64(p, x, y, z);
}
extern __device__ void g1p_fromG1a(g1p_t &p, const g1a_t &a);
extern __device__ __host__ void g1p_print(const char *s, const g1p_t &p);
extern __device__ __host__ void g1p_cpy(g1p_t &p, const g1p_t &q);

extern __device__  bool g1p_eq(const g1p_t &p, const g1p_t &q);
extern __device__  bool g1p_neq(const g1p_t &p, const g1p_t &q);
extern __device__  bool g1p_isInf(const g1p_t &p);
extern __device__  bool g1p_isPoint(const g1p_t &p);

extern __device__ void g1p_neg(g1p_t &p);
extern __device__ void g1p_scale(g1p_t &p, const fp_t &s);
extern __device__ void g1p_dbl(g1p_t &p);
extern __device__ void g1p_add(g1p_t &p, const g1p_t &q);
extern __device__ void g1p_sub(g1p_t &p, const g1p_t &q);
extern __device__ void g1p_addsub(g1p_t &p, g1p_t &q);
extern __device__ void g1p_mul(g1p_t &p, const fr_t &x);

extern __device__ __host__ void g1a_inf(g1a_t &a);
extern __device__ __host__ void g1a_gen(g1a_t &a);

extern __device__ __host__ void g1p_inf(g1p_t &p);
extern __device__ __host__ void g1p_gen(g1p_t &p);

// Device-side FFT functions

extern __device__ void g1p_fft(g1p_t *output, const g1p_t *input);
extern __device__ void g1p_ift(g1p_t *output, const g1p_t *input);

// Kernel wrappers for device-side FFT functions

__global__ void g1p_fft_wrapper(g1p_t *output, const g1p_t *input);
__global__ void g1p_ift_wrapper(g1p_t *output, const g1p_t *input);
__global__ void g1a_fromG1p_wrapper(g1a_t *a, size_t len, const g1p_t *p);

// Debug helpers

__global__ void g1p_eq_wrapper(uint8_t *output, size_t len, const g1p_t *p, const g1p_t *q);


//Old version for comparisons:
__device__ void g1p_dbl_noPTX(g1p_t &p);

#endif

// vim: ts=4 et sw=4 si
