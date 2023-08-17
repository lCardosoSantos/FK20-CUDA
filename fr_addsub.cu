// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"
#include "fr_add.cuh"
#include "fr_sub.cuh"

// (x,y) := (x+y,x-y)

/**
 * @brief Calculates the sum and the difference of the arguments, storing back into the arguments: (x,y) := (x+y,x-y). Device function.
 * 
 * @param[in,out] x First operand, will store the sum after execution.
 * @param[in,out] y Second operand, will store the difference after execution.
 * @return void 
 */
__device__ void fr_addsub(fr_t &x, fr_t &y) {
    unsigned tid = 0;   tid += blockIdx.z;
    tid *= gridDim.y;   tid += blockIdx.y;
    tid *= gridDim.x;   tid += blockIdx.x;
    tid *= blockDim.z;  tid += threadIdx.z;
    tid *= blockDim.y;  tid += threadIdx.y;
    tid *= blockDim.x;  tid += threadIdx.x;

    uint64_t
        x0 = x[0], y0 = y[0],
        x1 = x[1], y1 = y[1],
        x2 = x[2], y2 = y[2],
        x3 = x[3], y3 = y[3];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 t<4>, x<4>, y<4>;"
    "\n\t.reg .u32 t4;"
    "\n\t.reg .pred nz;"

    "\n\tmov.u64 x0, %0;"
    "\n\tmov.u64 x1, %1;"
    "\n\tmov.u64 x2, %2;"
    "\n\tmov.u64 x3, %3;"

    "\n\tmov.u64 y0, %4;"
    "\n\tmov.u64 y1, %5;"
    "\n\tmov.u64 y2, %6;"
    "\n\tmov.u64 y3, %7;"

    FR_ADD(t, x, y)
    FR_SUB(y, x, y)

    "\n\tmov.u64 %0, t0;"
    "\n\tmov.u64 %1, t1;"
    "\n\tmov.u64 %2, t2;"
    "\n\tmov.u64 %3, t3;"

    "\n\tmov.u64 %4, y0;"
    "\n\tmov.u64 %5, y1;"
    "\n\tmov.u64 %6, y2;"
    "\n\tmov.u64 %7, y3;"

    "\n\t}"
    :
    "+l"(x0), "+l"(x1), "+l"(x2), "+l"(x3),
    "+l"(y0), "+l"(y1), "+l"(y2), "+l"(y3)
    ); 

    x[0] = x0, x[1] = x1, x[2] = x2, x[3] = x3;
    y[0] = y0, y[1] = y1, y[2] = y2, y[3] = y3;
}

// vim: ts=4 et sw=4 si
