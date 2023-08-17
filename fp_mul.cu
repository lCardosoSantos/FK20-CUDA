// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fp_mul.cuh"
#include "fp_reduce12.cuh"

/**
 * @brief Multiplies two Fp residues, stores in z.
 *
 * @param[out] z
 * @param[in] x
 * @param[in] y
 * @return void
 */
__device__ void fp_mul(fp_t &z, const fp_t &x, const fp_t &y) {
    uint64_t
        x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5],
        y0 = y[0], y1 = y[1], y2 = y[2], y3 = y[3], y4 = y[4], y5 = y[5],
        z0, z1, z2, z3, z4, z5;

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 x<6>, y<6>;"
    "\n\t.reg .u64 u<10>, ua, ub;"
    "\n\t.reg .u64 q<8>;"
    "\n\t.reg .u64 r<7>;"

    "\n\tmov.u64 x0,  %6;"
    "\n\tmov.u64 x1,  %7;"
    "\n\tmov.u64 x2,  %8;"
    "\n\tmov.u64 x3,  %9;"
    "\n\tmov.u64 x4, %10;"
    "\n\tmov.u64 x5, %11;"

    "\n\tmov.u64 y0, %12;"
    "\n\tmov.u64 y1, %13;"
    "\n\tmov.u64 y2, %14;"
    "\n\tmov.u64 y3, %15;"
    "\n\tmov.u64 y4, %16;"
    "\n\tmov.u64 y5, %17;"

FP_MUL(u, x, y)
FP_REDUCE12(u)

    "\n\tmov.u64 %0,  u0;"
    "\n\tmov.u64 %1,  u1;"
    "\n\tmov.u64 %2,  u2;"
    "\n\tmov.u64 %3,  u3;"
    "\n\tmov.u64 %4,  u4;"
    "\n\tmov.u64 %5,  u5;"

    "\n\t}"
    :
    "=l"(z0), "=l"(z1), "=l"(z2), "=l"(z3), "=l"(z4), "=l"(z5)
    :
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5),
    "l"(y0), "l"(y1), "l"(y2), "l"(y3), "l"(y4), "l"(y5)
    );

    z[0] = z0; z[1] = z1; z[2] = z2; z[3] = z3; z[4] = z4; z[5] = z5;
}

// vim: ts=4 et sw=4 si
