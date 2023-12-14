// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FR_CUH
#define FR_CUH

#include <stdint.h>

/**
 * @brief Subgroup element stored as a 256-bit array (a 4-element little-endian array of uint64_t). fr_t[0] is the least significant element.
 *
 * This type is used for the BLS12-381 subgroup. This group is derived from the case
 * \f$ k \equiv 0 (mod 6) \f$ of Construction 6.6 in the taxonomy (eprint 2006/372),
 * which results in a parameter x = -0xd201000000010000. The subgroup size is given by
 * the equation  \f$ x^4 -x^2 +1 \f$ and is numerically 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001.
 *
 * Implementation-wise, the following constants are hardcoded and indicated, when used:
 *  r     - Modulus
 *  rmu   - Reciprocal of the modulus
 *  rmmu0 - Maximum integer multiple of the modulus such that rmmu0 < 2**256
 *  rmmu1 - Minimum integer multiple of the modulus such that rmmu1 >= 2**256
 */
typedef uint64_t fr_t[4];

/**
 * @brief Table for the precomputed root-of-unity values.
 *
 */
extern __constant__ fr_t fr_roots[515];
extern fr_t fr_roots_host[513];

extern __device__ __host__ void fr_fromUint64(fr_t &z, const uint64_t *x);
extern __device__ void fr_toUint64(const fr_t &x, uint64_t *z);
extern __device__ __host__ void fr_cpy(fr_t &z, const fr_t &x);
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
extern __device__ __host__ void fr_zero(fr_t &z);
extern __device__ __host__ void fr_one(fr_t &z);

extern __device__ bool fr_eq(const fr_t &x, const fr_t &y);
extern __device__ bool fr_neq(const fr_t &x, const fr_t &y);
extern __device__ bool fr_nonzero(const fr_t &x);
extern __device__ bool fr_iszero(const fr_t &x);
extern __device__ bool fr_isone(const fr_t &x);

extern __device__ void fr_print(const char *s, const fr_t &x);

// Device-side FFT functions

extern __device__ void fr_fft(fr_t *output, const fr_t *input);
extern __device__ void fr_ift(fr_t *output, const fr_t *input);

// Kernel wrappers for device-side FFT functions

__global__ void fr_fft_wrapper(fr_t *output, const fr_t *input);
__global__ void fr_ift_wrapper(fr_t *output, const fr_t *input);
__global__ void fr_eq_wrapper(uint8_t *eq, int count, const fr_t *x, const fr_t *y);

#endif

// vim: ts=4 et sw=4 si
