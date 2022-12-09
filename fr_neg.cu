// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"

__device__ void fr_neg(uint64_t *z) {
    uint64_t
        z0 = z[0],
        z1 = z[1],
        z2 = z[2],
        z3 = z[3];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u32 cf;"
    "\n\t.reg .pred bp;"

    // z = rmmu0 - z

    "\n\tsub.u64.cc  %0, 0xFFFFFFFE00000002U, %0;"
    "\n\tsubc.u64.cc %1, 0xA77B4805FFFCB7FDU, %1;"
    "\n\tsubc.u64.cc %2, 0x6673B0101343B00AU, %2;"
    "\n\tsubc.u64.cc %3, 0xE7DB4EA6533AFA90U, %3;"
    "\n\tsubc.u32    cf,  0, 0;" // store carry flag in u32
    "\n\tsetp.hi.u32 bp, cf, 0;" // store carry flag in borrow predicate

    // if borrow then z += r

    "\n@bp\tadd.u64.cc  %0, %0, 0xFFFFFFFF00000001U;"
    "\n@bp\taddc.u64.cc %1, %1, 0x53BDA402FFFE5BFEU;"
    "\n@bp\taddc.u64.cc %2, %2, 0x3339D80809A1D805U;"
    "\n@bp\taddc.u64    %3, %3, 0x73EDA753299D7D48U;"

    "\n\t}"
    :
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3)
    ); 

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3;
}

// vim: ts=4 et sw=4 si
