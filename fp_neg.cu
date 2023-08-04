// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"

// fp_neg: Compute an additive inverse of a residue x modulo p.
// Subtracts x from the highest multiple of p less than 2^384,
// then adds p in case of underflow.
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
    "\n\t.reg .u32 z6;"
    "\n\t.reg .pred nz;"

    // z = mmu0 - x

    "\n\tsub.u64.cc  %0, 0x89F6FFFFFFFD0003U,  %6;"
    "\n\tsubc.u64.cc %1, 0x140BFFF43BF3FFFDU,  %7;"
    "\n\tsubc.u64.cc %2, 0xA0B767A8AC38A745U,  %8;"
    "\n\tsubc.u64.cc %3, 0x8831A7AC8FADA8BAU,  %9;"
    "\n\tsubc.u64.cc %4, 0xA3F8E5685DA91392U, %10;"
    "\n\tsubc.u64.cc %5, 0xEA09A13C057F1B6CU, %11;"
    "\n\tsubc.u32    z6,  0, 0;"
    "\n\tsetp.ne.u32 nz, z6, 0;"

    // if nz (borrow) then z += p

    "\n@nz\tadd.u64.cc  %0, %0, 0xB9FEFFFFFFFFAAABU;"
    "\n@nz\taddc.u64.cc %1, %1, 0x1EABFFFEB153FFFFU;"
    "\n@nz\taddc.u64.cc %2, %2, 0x6730D2A0F6B0F624U;"
    "\n@nz\taddc.u64.cc %3, %3, 0x64774B84F38512BFU;"
    "\n@nz\taddc.u64.cc %4, %4, 0x4B1BA7B6434BACD7U;"
    "\n@nz\taddc.u64    %5, %5, 0x1A0111EA397FE69AU;"

    "\n\t}"
    :
    "=l"(z0), "=l"(z1), "=l"(z2), "=l"(z3), "=l"(z4), "=l"(z5)
    :
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5)
    ); 

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3, z[4] = z4, z[5] = z5;
}

// vim: ts=4 et sw=4 si
