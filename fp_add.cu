// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fp.cuh"
#include "fp_add.cuh"

__device__ void fp_add(fp_t &z, const fp_t &x, const fp_t &y) {
    uint64_t
        x0 = x[0], y0 = y[0], z0,
        x1 = x[1], y1 = y[1], z1,
        x2 = x[2], y2 = y[2], z2,
        x3 = x[3], y3 = y[3], z3,
        x4 = x[4], y4 = y[4], z4,
        x5 = x[5], y5 = y[5], z5;

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 z<6>, x<6>, y<6>;"
    "\n\t.reg .u32 t6;"
    "\n\t.reg .pred cp;"

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

FP_ADD(z, x, y)

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
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5),
    "l"(y0), "l"(y1), "l"(y2), "l"(y3), "l"(y4), "l"(y5)
    ); 

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3, z[4] = z4, z[5] = z5;
}

// vim: ts=4 et sw=4 si
