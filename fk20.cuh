// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FK20_CUH
#define FK20_CUH

#include <stdint.h>

__global__ void fk20_fft(g1p_t *output, const g1p_t *input);

#endif

// vim: ts=4 et sw=4 si
