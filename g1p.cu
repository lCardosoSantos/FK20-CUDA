// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <stdio.h>

#include "fp.cuh"
#include "g1.cuh"

/**
 * @brief Converts G1 point into arrays of uint64_t. Each array must be uint64_t[6]
 * 
 * @param[in] p point in G1 
 * @param[out] x Array with the x coordinate
 * @param[out] y Array with the y coordinate
 * @param[out] z Array with the z coordinate
 * @return void 
 */
__device__ void g1p_toUint64(const g1p_t &p, uint64_t *x, uint64_t *y, uint64_t *z) {
    fp_toUint64(p.x, x);
    fp_toUint64(p.y, y);
    fp_toUint64(p.z, z);
}

/**
 * @brief Converts arrays of uint64_t into a G1 point. Each array must be uint64_t[6]
 * 
 * @param[out] p point in G1 
 * @param[in] x Array with the x coordinate
 * @param[in] y Array with the y coordinate
 * @param[in] z Array with the z coordinate
 * @return void 
 */
__device__ __host__ void g1p_fromUint64(g1p_t &p, uint64_t *x, uint64_t *y, uint64_t *z) {
    fp_fromUint64(p.x, x);
    fp_fromUint64(p.y, y);
    fp_fromUint64(p.z, z);
}

/**
 * @brief Convert a point in affine coordinates to projective coordinates
 * 
 * @param[out] p point in projective coordinates 
 * @param[in] a point in affine coordinates
 * @return void 
 */
__device__ void g1p_fromG1a(g1p_t &p, const g1a_t &a) {
    if (fp_iszero(a.x) && fp_iszero(a.y)) {
        g1p_inf(p);
    }
    fp_fromUint64(p.x, a.x);
    fp_fromUint64(p.y, a.y);
    fp_one(p.z);
}

/**
 * @brief Copy from q into p
 * 
 * @param[out] p 
 * @param[in] q 
 * @return void 
 */
__device__ __host__ void g1p_cpy(g1p_t &p, const g1p_t &q) {
    fp_cpy(p.x, q.x);
    fp_cpy(p.y, q.y);
    fp_cpy(p.z, q.z);
}

/**
 * @brief Print a standard representation of p, preceded by the user-set string s
 * 
 * @param[out] s message string
 * @param[out] p point in g1
 * @return void 
 */
__device__ __host__ void g1p_print(const char *s, const g1p_t &p) {
    printf("%s ", s);
    printf("#x%016lx%016lx%016lx%016lx%016lx%016lx ",  p.x[5], p.x[4], p.x[3], p.x[2], p.x[1], p.x[0]);
    printf("#x%016lx%016lx%016lx%016lx%016lx%016lx ",  p.y[5], p.y[4], p.y[3], p.y[2], p.y[1], p.y[0]);
    printf("#x%016lx%016lx%016lx%016lx%016lx%016lx\n", p.z[5], p.z[4], p.z[3], p.z[2], p.z[1], p.z[0]);
}

/**
 * @brief Set p to the point-at-infinity (0,1,0)
 * 
 * @param[in] p 
 * @return void 
 */
__device__ __host__ void g1p_inf(g1p_t &p) {
    for (int i=0; i<6; i++)
        p.x[i] = p.y[i] = p.z[i] = 0;

    p.y[0] = 1;
};

/**
 * @brief Sets p to the generator point G1 of bls12_381
 * 
 * @param p 
 * @return Void 
 */
__device__ __host__ void g1p_gen(g1p_t &p) {
    p.x[5] = 0x17F1D3A73197D794;
    p.x[4] = 0x2695638C4FA9AC0F;
    p.x[3] = 0xC3688C4F9774B905;
    p.x[2] = 0xA14E3A3F171BAC58;
    p.x[1] = 0x6C55E83FF97A1AEF;
    p.x[0] = 0xFB3AF00ADB22C6BB;

    p.y[5] = 0x08B3F481E3AAA0F1;
    p.y[4] = 0xA09E30ED741D8AE4;
    p.y[3] = 0xFCF5E095D5D00AF6;
    p.y[2] = 0x00DB18CB2C04B3ED;
    p.y[1] = 0xD03CC744A2888AE4;
    p.y[0] = 0x0CAA232946C5E7E1;

    p.z[5] = 0;
    p.z[4] = 0;
    p.z[3] = 0;
    p.z[2] = 0;
    p.z[1] = 0;
    p.z[0] = 1;
};

// Kernel wrappers for device-side functions

/**
 * @brief Host-side comparison of arrays of g1p_t. 
 * 
 * @param[out] eq Array of size count, will stop the result of the comparison. eq[i]==1 if p[i]==q[i]
 * @param[in] count number of elements to be compared
 * @param[in] p Array g1p_t[count]
 * @param[in] q Array g1p_t[count]
 * @return void
 */
__global__ void g1p_eq_wrapper(uint8_t *eq, size_t count, const g1p_t *p, const g1p_t *q) {

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    __syncthreads();

    unsigned step = gridDim.z * gridDim.y * gridDim.x
                * blockDim.z * blockDim.y * blockDim.x;

    for (unsigned i=tid; i<count; i+=step)
        eq[i] = g1p_eq(p[i], q[i]) ? 1 : 0;
}

/**
 * @brief Host-side conversion of points in projective coordinates into affine coordinates.
 * 
 * @param[out] a Array g1a_t[count]
 * @param[in] count number of elements in the array
 * @param[in] p Array g1p_t[count]
 * @return void
 */
__global__ void g1a_fromG1p_wrapper(g1a_t *a, size_t count, const g1p_t *p) {

    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    unsigned step = gridDim.z * gridDim.y * gridDim.x
                * blockDim.z * blockDim.y * blockDim.x;

    for (unsigned i=tid; i<count; i+=step)
        g1a_fromG1p(*a, *p);
}

// vim: ts=4 et sw=4 si
