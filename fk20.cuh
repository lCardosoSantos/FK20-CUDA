// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FK20_CUH
#define FK20_CUH

#include <stdint.h>

__global__ void g1p_fft(g1p_t *output, const g1p_t *input);
__global__ void g1p_ift(g1p_t *output, const g1p_t *input);
__global__ void fr_fft(fr_t *output, const fr_t *input);
__global__ void fr_ift(fr_t *output, const fr_t *input);

#endif

// vim: ts=4 et sw=4 si
