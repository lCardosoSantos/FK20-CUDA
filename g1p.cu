// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "fp.cuh"
#include "g1.cuh"

__device__ void g1p_toUint64(uint64_t *p, uint64_t *x, uint64_t *y, uint64_t *z) {
    fp_toUint64(p+ 0, x);
    fp_toUint64(p+ 6, y);
    fp_toUint64(p+12, z);
}

__device__ void g1p_fromUint64(uint64_t *p, uint64_t *x, uint64_t *y, uint64_t *z) {
    fp_fromUint64(p+ 0, x);
    fp_fromUint64(p+ 6, y);
    fp_fromUint64(p+12, z);
}

__device__ void g1p_fromG1a(uint64_t *p, const uint64_t *a) {
    if (fp_iszero(a+0) && fp_iszero(a+6)) {
        g1p_inf(p);
    }
    fp_fromUint64(p+ 0, a+ 0);
    fp_fromUint64(p+ 6, a+ 6);
    fp_one(p+12);
}

__device__ void g1p_cpy(uint64_t *z, const uint64_t *x) {
    for (int i=0; i<18; i++)
        z[i] = x[i];
}

__device__ void g1p_print(const char *s, const uint64_t *p) {
    printf("%s ", s);
    printf("0x%016lx%016lx%016lx%016lx%016lx%016lx ",  p[ 5], p[ 4], p[ 3], p[ 2], p[ 1], p[ 0]);
    printf("0x%016lx%016lx%016lx%016lx%016lx%016lx ",  p[11], p[10], p[ 9], p[ 8], p[ 7], p[ 6]);
    printf("0x%016lx%016lx%016lx%016lx%016lx%016lx\n", p[17], p[16], p[15], p[14], p[13], p[12]);
}

__device__ void g1p_inf(uint64_t *p) {
    for (int i=0; i<18; i++)
        p[i] = 0;

    p[6] = 1;
};

__device__ void g1p_gen(uint64_t *p) {
    p[ 0] = 0xFB3AF00ADB22C6BB;
    p[ 1] = 0x6C55E83FF97A1AEF;
    p[ 2] = 0xA14E3A3F171BAC58;
    p[ 3] = 0xC3688C4F9774B905;
    p[ 4] = 0x2695638C4FA9AC0F;
    p[ 5] = 0x17F1D3A73197D794;

    p[ 6] = 0x0CAA232946C5E7E1;
    p[ 7] = 0xD03CC744A2888AE4;
    p[ 8] = 0x00DB18CB2C04B3ED;
    p[ 9] = 0xFCF5E095D5D00AF6;
    p[10] = 0xA09E30ED741D8AE4;
    p[11] = 0x08B3F481E3AAA0F1;

    p[12] = 1;
    p[13] = 0;
    p[14] = 0;
    p[15] = 0;
    p[16] = 0;
    p[17] = 0;
};

// vim: ts=4 et sw=4 si
