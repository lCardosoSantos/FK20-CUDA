// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "fr_add.cuh"

/**
 * @brief Accumulates x in z. Device only function.
 * 
 * @param[in, out] z 
 * @param[in ] x 
 * @return void
 */
__device__ void fr_add(fr_t &z, const fr_t &x) {
    uint64_t
        z0 = z[0], x0 = x[0],
        z1 = z[1], x1 = x[1],
        z2 = z[2], x2 = x[2],
        z3 = z[3], x3 = x[3];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 x<4>, z<4>;"
    "\n\t.reg .u32 t4;"
    "\n\t.reg .pred nz;"

    "\n\tmov.u64 z0, %0;"
    "\n\tmov.u64 z1, %1;"
    "\n\tmov.u64 z2, %2;"
    "\n\tmov.u64 z3, %3;"

    "\n\tmov.u64 x0, %4;"
    "\n\tmov.u64 x1, %5;"
    "\n\tmov.u64 x2, %6;"
    "\n\tmov.u64 x3, %7;"

FR_ADD(z, z, x)

    "\n\tmov.u64 %0, z0;"
    "\n\tmov.u64 %1, z1;"
    "\n\tmov.u64 %2, z2;"
    "\n\tmov.u64 %3, z3;"

    "\n\t}"
    :
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3)
    :
    "l"(x0), "l"(x1), "l"(x2), "l"(x3)
    ); 

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3;
}

// vim: ts=4 et sw=4 si
