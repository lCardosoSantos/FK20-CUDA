// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include <stdio.h>

#include "fp.cuh"
#include "fr.cuh"
#include "g1.cuh"

__device__ void g1a_fromUint64(g1a_t &a, const uint64_t *x, const uint64_t *y) {
    fp_fromUint64(a.x, x);
    fp_fromUint64(a.y, y);
}

__device__ void g1a_fromFp(g1a_t &a, const fp_t &x, const fp_t &y) {
    fp_cpy(a.x, x);
    fp_cpy(a.y, y);
}

__device__ void g1a_fromG1p(g1a_t &a, const g1p_t &p) {
    fp_t inv;

    fp_inv(inv, p.z);

    fp_mul(a.x, p.x, inv);
    fp_mul(a.y, p.y, inv);
}

__device__ void g1a_cpy(g1a_t &a, const g1a_t &b) {
    fp_cpy(a.x, b.x);
    fp_cpy(a.y, b.y);
}

__device__ void g1a_print(const char *s, const g1a_t &a) {
    printf("%s ", s);
    printf("0x%016lx%016lx%016lx%016lx%016lx%016lx ",  a.x[5], a.x[4], a.x[3], a.x[2], a.x[1], a.x[0]);
    printf("0x%016lx%016lx%016lx%016lx%016lx%016lx\n", a.y[5], a.y[4], a.y[3], a.y[2], a.y[1], a.y[0]);
}


__device__ void g1a_inf(g1a_t &a) {
    fp_zero(a.x);
    fp_zero(a.y);
};

__device__ void g1a_gen(g1a_t &a) {
    a.x[5] = 0x17F1D3A73197D794;
    a.x[4] = 0x2695638C4FA9AC0F;
    a.x[3] = 0xC3688C4F9774B905;
    a.x[2] = 0xA14E3A3F171BAC58;
    a.x[1] = 0x6C55E83FF97A1AEF;
    a.x[0] = 0xFB3AF00ADB22C6BB;

    a.y[5] = 0x08B3F481E3AAA0F1;
    a.y[4] = 0xA09E30ED741D8AE4;
    a.y[3] = 0xFCF5E095D5D00AF6;
    a.y[2] = 0x00DB18CB2C04B3ED;
    a.y[1] = 0xD03CC744A2888AE4;
    a.y[0] = 0x0CAA232946C5E7E1;
};

// vim: ts=4 et sw=4 si
