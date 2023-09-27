
// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_PTX_CUH
#define FP_PTX_CUH

#include "fp.cuh"


extern __device__ void fp_add_ptx(fp_t &z, const fp_t &x, const fp_t &y);
extern __device__ void fp_sub_ptx(fp_t &z, const fp_t &x, const fp_t &y);
extern __device__ void fp_mul_ptx(fp_t &z, const fp_t &x, const fp_t &y);
extern __device__ void fp_sqr_ptx(fp_t &z, const fp_t &x);

extern __device__ void fp_x2_ptx(fp_t &z, const fp_t &x);
extern __device__ void fp_x3_ptx(fp_t &z, const fp_t &x);
extern __device__ void fp_x4_ptx(fp_t &z, const fp_t &x);
extern __device__ void fp_x8_ptx(fp_t &z, const fp_t &x);
extern __device__ void fp_x12_ptx(fp_t &z, const fp_t &x);
#endif
