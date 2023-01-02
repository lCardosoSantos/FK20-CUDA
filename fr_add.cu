// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"

__device__ void fr_add(fr_t &z, const fr_t &x) {
    uint64_t
        z0 = z[0], x0 = x[0],
        z1 = z[1], x1 = x[1],
        z2 = z[2], x2 = x[2],
        z3 = z[3], x3 = x[3];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 t0, t1, t2, t3;"
    "\n\t.reg .u32 cf;"
    "\n\t.reg .pred cp, bp;"

    // z,c = z + x

    "\n\tadd.u64.cc  %0, %0,  %4;"
    "\n\taddc.u64.cc %1, %1,  %5;"
    "\n\taddc.u64.cc %2, %2,  %6;"
    "\n\taddc.u64.cc %3, %3,  %7;"
    "\n\taddc.u32    cf,  0, 0;" // store carry flag in u32
    "\n\tsetp.ne.u32 cp, cf, 0;" // store carry flag in carry predicate

    // if carry then z -= rmmu0

    "\n@cp\tsub.u64.cc  %0, %0, 0xFFFFFFFE00000002U;"
    "\n@cp\tsubc.u64.cc %1, %1, 0xA77B4805FFFCB7FDU;"
    "\n@cp\tsubc.u64.cc %2, %2, 0x6673B0101343B00AU;"
    "\n@cp\tsubc.u64.cc %3, %3, 0xE7DB4EA6533AFA90U;"
    "\n@cp\tsubc.u32    cf,  0, 0;"
    "\n@cp\tsetp.eq.and.u32 cp, cf, 0, cp;" // cp = cp & (cf ? 0 : 1)

    // if not borrow then z -= r

    "\n@cp\tsub.u64.cc  %0, %0, 0xFFFFFFFF00000001U;"
    "\n@cp\tsubc.u64.cc %1, %1, 0x53BDA402FFFE5BFEU;"
    "\n@cp\tsubc.u64.cc %2, %2, 0x3339D80809A1D805U;"
    "\n@cp\tsubc.u64    %3, %3, 0x73EDA753299D7D48U;"

    "\n\t}"
    :
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3),
    "+l"(x0), "+l"(x1), "+l"(x2), "+l"(x3)
    ); 

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3;
}

// vim: ts=4 et sw=4 si
