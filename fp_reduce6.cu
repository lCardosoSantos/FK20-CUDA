// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"

__device__ void fp_reduce6(fp_t &z) {
    uint64_t
        z0 = z[0],
        z1 = z[1],
        z2 = z[2],
        z3 = z[3],
        z4 = z[4],
        z5 = z[5];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 t<7>, q5;"
    "\n\t.reg .pred nz;"

    // q1 = x/2^320; q2 = q1 * mu; q3 = q2 / 2^448

    "\n\tmul.hi.u64     q5, %5,  9;"    // mu6 == 9

    // r2 = q3 * m mod 2^448

    "\n\tmul.lo.u64     t0, q5, 0xB9FEFFFFFFFFAAABU;"   // p0
    "\n\tmul.hi.u64     t1, q5, 0xB9FEFFFFFFFFAAABU;"

    "\n\tmul.lo.u64     t2, q5, 0x6730D2A0F6B0F624U;"   // p2
    "\n\tmul.hi.u64     t3, q5, 0x6730D2A0F6B0F624U;"

    "\n\tmul.lo.u64     t4, q5, 0x4B1BA7B6434BACD7U;"   // p4
    "\n\tmul.hi.u64     t5, q5, 0x4B1BA7B6434BACD7U;"


    "\n\tmad.lo.u64.cc  t1, q5, 0x1EABFFFEB153FFFFU, t1;"   // p1
    "\n\tmadc.hi.u64.cc t2, q5, 0x1EABFFFEB153FFFFU, t2;"

    "\n\tmadc.lo.u64.cc t3, q5, 0x64774B84F38512BFU, t3;"   // p3
    "\n\tmadc.hi.u64.cc t4, q5, 0x64774B84F38512BFU, t4;"

    "\n\tmadc.lo.u64.cc t5, q5, 0x1A0111EA397FE69AU, t5;"   // p5
    "\n\tmadc.hi.u64.cc t6, q5, 0x1A0111EA397FE69AU, 0;"

    // r = r1 - r2 = z - r2

    // Note: x < 2^384
    //      => q3 <= x/m
    //      => q3*m <= x
    //      => r2 <= x
    //      => r >= 0

    "\n\tsub.u64.cc     %0, %0, t0;"
    "\n\tsubc.u64.cc    %1, %1, t1;"
    "\n\tsubc.u64.cc    %2, %2, t2;"
    "\n\tsubc.u64.cc    %3, %3, t3;"
    "\n\tsubc.u64.cc    %4, %4, t4;"
    "\n\tsubc.u64       %5, %5, t5;"

    "\n\tsub.u64.cc     t0, %0, 0xB9FEFFFFFFFFAAABU;"
    "\n\tsubc.u64.cc    t1, %1, 0x1EABFFFEB153FFFFU;"
    "\n\tsubc.u64.cc    t2, %2, 0x6730D2A0F6B0F624U;"
    "\n\tsubc.u64.cc    t3, %3, 0x64774B84F38512BFU;"
    "\n\tsubc.u64.cc    t4, %4, 0x4B1BA7B6434BACD7U;"
    "\n\tsubc.u64.cc    t5, %5, 0x1A0111EA397FE69AU;"
    "\n\tsubc.u64       t6,  0, 0;"
    "\n\tsetp.ne.u64    nz, t6, 0;"

    "\n@!nz\tmov.u64    %0, t0;"
    "\n@!nz\tmov.u64    %1, t1;"
    "\n@!nz\tmov.u64    %2, t2;"
    "\n@!nz\tmov.u64    %3, t3;"
    "\n@!nz\tmov.u64    %4, t4;"
    "\n@!nz\tmov.u64    %5, t5;"

    "\n\tsub.u64.cc     t0, %0, 0xB9FEFFFFFFFFAAABU;"
    "\n\tsubc.u64.cc    t1, %1, 0x1EABFFFEB153FFFFU;"
    "\n\tsubc.u64.cc    t2, %2, 0x6730D2A0F6B0F624U;"
    "\n\tsubc.u64.cc    t3, %3, 0x64774B84F38512BFU;"
    "\n\tsubc.u64.cc    t4, %4, 0x4B1BA7B6434BACD7U;"
    "\n\tsubc.u64.cc    t5, %5, 0x1A0111EA397FE69AU;"
    "\n\tsubc.u64       t6,  0, 0;"
    "\n\tsetp.ne.u64    nz, t6, 0;"

    "\n@!nz\tmov.u64    %0, t0;"
    "\n@!nz\tmov.u64    %1, t1;"
    "\n@!nz\tmov.u64    %2, t2;"
    "\n@!nz\tmov.u64    %3, t3;"
    "\n@!nz\tmov.u64    %4, t4;"
    "\n@!nz\tmov.u64    %5, t5;"

    "\n\t}"
    :
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3), "+l"(z4), "+l"(z5)
    ); 

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3, z[4] = z4, z[5] = z5;
}

// vim: ts=4 et sw=4 si
