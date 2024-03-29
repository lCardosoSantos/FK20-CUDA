// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fp_neg.cuh"

/**
 * @brief Compute an additive inverse of a residue x modulo p. Z ← -X
 * Subtracts x from the highest multiple of p less than 2^384,
 * then adds p in case of underflow.
 *
 * @param[out] z
 * @param[in] x
 * @return void
 */
__device__ void fp_neg(fp_t &z, const fp_t &x) {
    uint64_t
        x0 = x[0], z0,
        x1 = x[1], z1,
        x2 = x[2], z2,
        x3 = x[3], z3,
        x4 = x[4], z4,
        x5 = x[5], z5;

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 z<7>, x<6>;"
    "\n\t.reg .pred nz;"

    "\n\tmov.u64 x0,  %6;"
    "\n\tmov.u64 x1,  %7;"
    "\n\tmov.u64 x2,  %8;"
    "\n\tmov.u64 x3,  %9;"
    "\n\tmov.u64 x4, %10;"
    "\n\tmov.u64 x5, %11;"

FP_NEG(z, x)

    "\n\tmov.u64 %0,  z0;"
    "\n\tmov.u64 %1,  z1;"
    "\n\tmov.u64 %2,  z2;"
    "\n\tmov.u64 %3,  z3;"
    "\n\tmov.u64 %4,  z4;"
    "\n\tmov.u64 %5,  z5;"

    "\n\t}"
    :
    "=l"(z0), "=l"(z1), "=l"(z2), "=l"(z3), "=l"(z4), "=l"(z5)
    :
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5)
    );

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3, z[4] = z4, z[5] = z5;
}

// vim: ts=4 et sw=4 si
