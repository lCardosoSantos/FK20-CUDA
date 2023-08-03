// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"

/**
 * @brief Reduced the value in fr_t to the field modulus
 * 
 * Modulus = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001 
 * and is hardcoded in the function.
 * 
 * @param z 
 * @return void 
 */
__device__ void fr_reduce4(fr_t &z) {
    uint64_t
        z0 = z[0],
        z1 = z[1],
        z2 = z[2],
        z3 = z[3];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 t<5>;"
    "\n\t.reg .pred cp;"

    // If z > 2^192*floor(r/2^192), then z -= r

    "\n\tsetp.gt.u64    cp, %3, 0x73EDA753299D7D48U;"

    "\n@cp\tsub.u64.cc  %0, %0, 0xFFFFFFFF00000001U;"
    "\n@cp\tsubc.u64.cc %1, %1, 0x53BDA402FFFE5BFEU;"
    "\n@cp\tsubc.u64.cc %2, %2, 0x3339D80809A1D805U;"
    "\n@cp\tsubc.u64.cc %3, %3, 0x73EDA753299D7D48U;"

    // t = z - r

    "\n\tsub.u64.cc     t0, %0, 0xFFFFFFFF00000001U;"
    "\n\tsubc.u64.cc    t1, %1, 0x53BDA402FFFE5BFEU;"
    "\n\tsubc.u64.cc    t2, %2, 0x3339D80809A1D805U;"
    "\n\tsubc.u64.cc    t3, %3, 0x73EDA753299D7D48U;"
    "\n\tsubc.u64       t4,  0, 0;"

    // If no underflow, then z = t

    "\n\tsetp.eq.u64    cp, t4, 0;"

    "\n@cp\tmov.u64     %0, t0;"
    "\n@cp\tmov.u64     %1, t1;"
    "\n@cp\tmov.u64     %2, t2;"
    "\n@cp\tmov.u64     %3, t3;"

    "\n\t}"
    :
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3)
    ); 

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3;
}

// vim: ts=4 et sw=4 si
