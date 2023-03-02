// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik

#include "fr.cuh"

// (x,y) := (x+y,x-y)

__device__ void fr_addsub(fr_t &x, fr_t &y) {
    uint64_t
        x0 = x[0], y0 = y[0],
        x1 = x[1], y1 = y[1],
        x2 = x[2], y2 = y[2],
        x3 = x[3], y3 = y[3];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 t<4>;"
    "\n\t.reg .u32 cf;"
    "\n\t.reg .pred cp, bp;"

    // t,b = x - y

    "\n\tsub.u64.cc  t0, %0,  %4;"
    "\n\tsubc.u64.cc t1, %1,  %5;"
    "\n\tsubc.u64.cc t2, %2,  %6;"
    "\n\tsubc.u64.cc t3, %3,  %7;"
    "\n\tsubc.u32    cf,  0, 0;" // store carry flag in u32
    "\n\tsetp.ne.u32 bp, cf, 0;" // store carry flag in borrow predicate

    // if borrow then t += rmmu0

    "\n@bp\tadd.u64.cc  t0, t0, 0xFFFFFFFE00000002U;"
    "\n@bp\taddc.u64.cc t1, t1, 0xA77B4805FFFCB7FDU;"
    "\n@bp\taddc.u64.cc t2, t2, 0x6673B0101343B00AU;"
    "\n@bp\taddc.u64.cc t3, t3, 0xE7DB4EA6533AFA90U;"
    "\n@bp\taddc.u32    cf,  0, 0;"
    "\n@bp\tsetp.eq.and.u32 bp, cf, 0, bp;" // bp = bp & (cf ? 0 : 1)

    // if not carry then t += r

    "\n@bp\tadd.u64.cc  t0, t0, 0xFFFFFFFF00000001U;"
    "\n@bp\taddc.u64.cc t1, t1, 0x53BDA402FFFE5BFEU;"
    "\n@bp\taddc.u64.cc t2, t2, 0x3339D80809A1D805U;"
    "\n@bp\taddc.u64    t3, t3, 0x73EDA753299D7D48U;"

    // x,c = x + y

    "\n\tadd.u64.cc  %0, %0,  %4;"
    "\n\taddc.u64.cc %1, %1,  %5;"
    "\n\taddc.u64.cc %2, %2,  %6;"
    "\n\taddc.u64.cc %3, %3,  %7;"
    "\n\taddc.u32    cf,  0, 0;" // store carry flag in u32
    "\n\tsetp.ne.u32 cp, cf, 0;" // store carry flag in carry predicate

    // if carry then x -= rmmu0

    "\n@cp\tsub.u64.cc  %0, %0, 0xFFFFFFFE00000002U;"
    "\n@cp\tsubc.u64.cc %1, %1, 0xA77B4805FFFCB7FDU;"
    "\n@cp\tsubc.u64.cc %2, %2, 0x6673B0101343B00AU;"
    "\n@cp\tsubc.u64.cc %3, %3, 0xE7DB4EA6533AFA90U;"
    "\n@cp\tsubc.u32    cf,  0, 0;"
    "\n@cp\tsetp.eq.and.u32 cp, cf, 0, cp;" // cp = cp & (cf ? 0 : 1)

    // if not borrow then x -= r

    "\n@cp\tsub.u64.cc  %0, %0, 0xFFFFFFFF00000001U;"
    "\n@cp\tsubc.u64.cc %1, %1, 0x53BDA402FFFE5BFEU;"
    "\n@cp\tsubc.u64.cc %2, %2, 0x3339D80809A1D805U;"
    "\n@cp\tsubc.u64    %3, %3, 0x73EDA753299D7D48U;"

    // y = t

    "\n@bp\tmov.u64 %4, t0;"
    "\n@bp\tmov.u64 %5, t1;"
    "\n@bp\tmov.u64 %6, t2;"
    "\n@bp\tmov.u64 %7, t3;"

    "\n\t}"
    :
    "+l"(x0), "+l"(x1), "+l"(x2), "+l"(x3),
    "+l"(y0), "+l"(y1), "+l"(y2), "+l"(y3)
    ); 

    x[0] = x0, x[1] = x1, x[2] = x2, x[3] = x3;
    y[0] = y0, y[1] = y1, y[2] = y2, y[3] = y3;
}

// vim: ts=4 et sw=4 si
