// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"

__device__ void fr_x4(uint64_t *z) {
    uint64_t
        z0 = z[0],
        z1 = z[1],
        z2 = z[2],
        z3 = z[3];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u32 t4;"
    "\n\t.reg .pred cp;"

    // z = z + z

    "\n\tadd.u64.cc  %0, %0, %0;"
    "\n\taddc.u64.cc %1, %1, %1;"
    "\n\taddc.u64.cc %2, %2, %2;"
    "\n\taddc.u64.cc %3, %3, %3;"
    "\n\taddc.u32    t4,  0,  0;"

    // if z >= 2^256 then z -= mmu0

    "\n\tsetp.ge.u32 cp, t4, 1;"
    "\n@cp\tsub.u64.cc  %0, %0, 0xFFFFFFFE00000002U;"
    "\n@cp\tsubc.u64.cc %1, %1, 0xA77B4805FFFCB7FDU;"
    "\n@cp\tsubc.u64.cc %2, %2, 0x6673B0101343B00AU;"
    "\n@cp\tsubc.u64.cc %3, %3, 0xE7DB4EA6533AFA90U;"
    "\n@cp\tsubc.u32    t4, t4, 0;"

    // z = z + z

    "\n\tadd.u64.cc  %0, %0, %0;"
    "\n\taddc.u64.cc %1, %1, %1;"
    "\n\taddc.u64.cc %2, %2, %2;"
    "\n\taddc.u64.cc %3, %3, %3;"
    "\n\taddc.u32    t4, t4, t4;"

    // if z >= 2^256 then z -= mmu0

    "\n\tsetp.ge.u32 cp, t4, 1;"
    "\n@cp\tsub.u64.cc  %0, %0, 0xFFFFFFFE00000002U;"
    "\n@cp\tsubc.u64.cc %1, %1, 0xA77B4805FFFCB7FDU;"
    "\n@cp\tsubc.u64.cc %2, %2, 0x6673B0101343B00AU;"
    "\n@cp\tsubc.u64.cc %3, %3, 0xE7DB4EA6533AFA90U;"
    "\n@cp\tsubc.u32    t4, t4, 0;"

    // if z >= 2^256 then z -= mmu0

    "\n\tsetp.ge.u32 cp, t4, 1;"
    "\n@cp\tsub.u64.cc  %0, %0, 0xFFFFFFFE00000002U;"
    "\n@cp\tsubc.u64.cc %1, %1, 0xA77B4805FFFCB7FDU;"
    "\n@cp\tsubc.u64.cc %2, %2, 0x6673B0101343B00AU;"
    "\n@cp\tsubc.u64    %3, %3, 0xE7DB4EA6533AFA90U;"

    "\n\t}"
    :
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3)); 

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3;
}

// vim: ts=4 et sw=4 si
