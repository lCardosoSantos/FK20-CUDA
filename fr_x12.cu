// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fr.cuh"

/**
 * @brief Multiply the subgroup element z by 12 with weak reduction
 * 
 * @param[in,out] z 
 * @return void 
 */
__device__ void fr_x12(fr_t &z) {
    uint64_t
        z0 = z[0],
        z1 = z[1],
        z2 = z[2],
        z3 = z[3];

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 t<4>;"
    "\n\t.reg .u32 t4;"
    "\n\t.reg .pred gt;"

    // t = z + z

    "\n\tadd.u64.cc  t0, %0, %0;"
    "\n\taddc.u64.cc t1, %1, %1;"
    "\n\taddc.u64.cc t2, %2, %2;"
    "\n\taddc.u64.cc t3, %3, %3;"
    "\n\taddc.u32    t4,  0,  0;"

    // z = z + t

    "\n\tadd.u64.cc  %0, %0, t0;"
    "\n\taddc.u64.cc %1, %1, t1;"
    "\n\taddc.u64.cc %2, %2, t2;"
    "\n\taddc.u64.cc %3, %3, t3;"
    "\n\taddc.u32    t4,  0, t4;"

    // z = z + z

    "\n\tadd.u64.cc  %0, %0, %0;"
    "\n\taddc.u64.cc %1, %1, %1;"
    "\n\taddc.u64.cc %2, %2, %2;"
    "\n\taddc.u64.cc %3, %3, %3;"
    "\n\taddc.u32    t4, t4, t4;"

    // z = z + z

    "\n\tadd.u64.cc  %0, %0, %0;"
    "\n\taddc.u64.cc %1, %1, %1;"
    "\n\taddc.u64.cc %2, %2, %2;"
    "\n\taddc.u64.cc %3, %3, %3;"
    "\n\taddc.u32    t4, t4, t4;"

    // if z >= 2^259 then z -= 17m

    "\n\tsetp.gt.u32 gt, t4, 7;"
    "\n@gt\tsub.u64.cc  %0, %0, 0xFFFFFFEF00000011U;"
    "\n@gt\tsubc.u64.cc %1, %1, 0x8F97E432FFE41BEEU;"
    "\n@gt\tsubc.u64.cc %2, %2, 0x66D75888A3BF585AU;"
    "\n@gt\tsubc.u64.cc %3, %3, 0xB2C81C85C37551CBU;"
    "\n@gt\tsubc.u32    t4, t4, 7;"

    // if z >= 2^258 then z -= 8m

    "\n\tsetp.gt.u32 gt, t4, 3;"
    "\n@gt\tsub.u64.cc  %0, %0, 0xFFFFFFF800000008U;"
    "\n@gt\tsubc.u64.cc %1, %1, 0x9DED2017FFF2DFF7U;"
    "\n@gt\tsubc.u64.cc %2, %2, 0x99CEC0404D0EC02AU;"
    "\n@gt\tsubc.u64.cc %3, %3, 0x9F6D3A994CEBEA41U;"
    "\n@gt\tsubc.u32    t4, t4, 3;"

    // if z >= 2^257 then z -= 4m

    "\n\tsetp.gt.u32 gt, t4, 1;"
    "\n@gt\tsub.u64.cc  %0, %0, 0xFFFFFFFC00000004U;"
    "\n@gt\tsubc.u64.cc %1, %1, 0x4EF6900BFFF96FFBU;"
    "\n@gt\tsubc.u64.cc %2, %2, 0xCCE7602026876015U;"
    "\n@gt\tsubc.u64.cc %3, %3, 0xCFB69D4CA675F520U;"
    "\n@gt\tsubc.u32    t4, t4, 1;"

    // if z >= 2^256 then z -= 2m

    "\n\tsetp.gt.u32 gt, t4, 0;"
    "\n@gt\tsub.u64.cc  %0, %0, 0xFFFFFFFE00000002U;"
    "\n@gt\tsubc.u64.cc %1, %1, 0xA77B4805FFFCB7FDU;"
    "\n@gt\tsubc.u64.cc %2, %2, 0x6673B0101343B00AU;"
    "\n@gt\tsubc.u64.cc %3, %3, 0xE7DB4EA6533AFA90U;"
    "\n@gt\tsubc.u32    t4, t4, 0;"

    // if z >= 2^256 then z -= 2m

    "\n\tsetp.gt.u32 gt, t4, 0;"
    "\n@gt\tsub.u64.cc  %0, %0, 0xFFFFFFFE00000002U;"
    "\n@gt\tsubc.u64.cc %1, %1, 0xA77B4805FFFCB7FDU;"
    "\n@gt\tsubc.u64.cc %2, %2, 0x6673B0101343B00AU;"
    "\n@gt\tsubc.u64    %3, %3, 0xE7DB4EA6533AFA90U;"

    "\n\t}"
    :
    "+l"(z0), "+l"(z1), "+l"(z2), "+l"(z3)); 

    z[0] = z0, z[1] = z1, z[2] = z2, z[3] = z3;
}

// vim: ts=4 et sw=4 si
