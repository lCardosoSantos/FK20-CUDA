// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fp_reduce7.cuh"

__device__ __noinline__ void fp_reduce7(
    fp_t &z,
    uint64_t x0,
    uint64_t x1,
    uint64_t x2,
    uint64_t x3,
    uint64_t x4,
    uint64_t x5,
    uint64_t x6
    )
{
    fp_reduce7(
        z[0], z[1], z[2], z[3], z[4], z[5],
        x0, x1, x2, x3, x4, x5, x6
    );
}
