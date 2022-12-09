// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "fp.cuh"
#include "fr.cuh"
#include "g1.cuh"

__device__ void g1a_fromUint64(uint64_t *a, const uint64_t *x, const uint64_t *y) {
    fp_cpy(a+0, x);
    fp_cpy(a+6, y);
}

__device__ void g1a_fromFp(uint64_t *a, const uint64_t *x, const uint64_t *y) {
    fp_cpy(a+0, x);
    fp_cpy(a+6, y);
}

__device__ void g1a_fromG1p(uint64_t *a, const uint64_t *p) {
    uint64_t inv[6];

    fp_inv(inv, p+12);

    fp_mul(a+0, a+0, inv);
    fp_mul(a+6, a+6, inv);
}

__device__ void g1a_cpy(uint64_t *a, const uint64_t *b) {
    for (int i=0; i<12; i++)
        a[i] = b[i];
}

__device__ void g1a_print(const char *s, const uint64_t *a) {
    printf("%s ", s);
    printf("0lx%016lx%016lx%016lx%016lx%016lx%016lx ",  a[ 5], a[ 4], a[ 3], a[ 2], a[ 1], a[ 0]);
    printf("0lx%016lx%016lx%016lx%016lx%016lx%016lx\n", a[11], a[10], a[ 9], a[ 8], a[ 7], a[ 6]);
}


__device__ void g1a_inf(uint64_t *a) {
    for (int i=0; i<12; i++)
        a[i] = 0;
};

__device__ void g1a_gen(uint64_t *a) {
    a[ 0] = 0xFB3AF00ADB22C6BB;
    a[ 1] = 0x6C55E83FF97A1AEF;
    a[ 2] = 0xA14E3A3F171BAC58;
    a[ 3] = 0xC3688C4F9774B905;
    a[ 4] = 0x2695638C4FA9AC0F;
    a[ 5] = 0x17F1D3A73197D794;

    a[ 6] = 0x0CAA232946C5E7E1;
    a[ 7] = 0xD03CC744A2888AE4;
    a[ 8] = 0x00DB18CB2C04B3ED;
    a[ 9] = 0xFCF5E095D5D00AF6;
    a[10] = 0xA09E30ED741D8AE4;
    a[11] = 0x08B3F481E3AAA0F1;
};

// vim: ts=4 et sw=4 si
