// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_X12
/**
 * @brief PTX macro for multiplication by 12. Stores in Z.
 *
 * Overwrites t0..t5 and z6.
 * Z and X may be the same.
 */
#define FP_X12(Z, X, T) \
    /* t = x + x */ \
 \
    "\n\tadd.u64.cc  "#T"0, "#X"0, "#X"0;" \
    "\n\taddc.u64.cc "#T"1, "#X"1, "#X"1;" \
    "\n\taddc.u64.cc "#T"2, "#X"2, "#X"2;" \
    "\n\taddc.u64.cc "#T"3, "#X"3, "#X"3;" \
    "\n\taddc.u64.cc "#T"4, "#X"4, "#X"4;" \
    "\n\taddc.u64.cc "#T"5, "#X"5, "#X"5;" \
    "\n\taddc.u64    "#Z"6,     0,     0;" \
 \
    /* z = x + t */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#X"0, "#T"0;" \
    "\n\taddc.u64.cc "#Z"1, "#X"1, "#T"1;" \
    "\n\taddc.u64.cc "#Z"2, "#X"2, "#T"2;" \
    "\n\taddc.u64.cc "#Z"3, "#X"3, "#T"3;" \
    "\n\taddc.u64.cc "#Z"4, "#X"4, "#T"4;" \
    "\n\taddc.u64.cc "#Z"5, "#X"5, "#T"5;" \
    "\n\taddc.u64    "#Z"6, "#Z"6,     0;" \
 \
    /* z = z + z */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#Z"0, "#Z"0;" \
    "\n\taddc.u64.cc "#Z"1, "#Z"1, "#Z"1;" \
    "\n\taddc.u64.cc "#Z"2, "#Z"2, "#Z"2;" \
    "\n\taddc.u64.cc "#Z"3, "#Z"3, "#Z"3;" \
    "\n\taddc.u64.cc "#Z"4, "#Z"4, "#Z"4;" \
    "\n\taddc.u64.cc "#Z"5, "#Z"5, "#Z"5;" \
    "\n\taddc.u64    "#Z"6, "#Z"6, "#Z"6;" \
 \
    /* z = z + z */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#Z"0, "#Z"0;" \
    "\n\taddc.u64.cc "#Z"1, "#Z"1, "#Z"1;" \
    "\n\taddc.u64.cc "#Z"2, "#Z"2, "#Z"2;" \
    "\n\taddc.u64.cc "#Z"3, "#Z"3, "#Z"3;" \
    "\n\taddc.u64.cc "#Z"4, "#Z"4, "#Z"4;" \
    "\n\taddc.u64.cc "#Z"5, "#Z"5, "#Z"5;" \
    "\n\taddc.u64    "#Z"6, "#Z"6, "#Z"6;" \
 \
    /* if z >= 2^387 then z -= 78m */ \
 \
    "\n\tsetp.gt.u64 gt, "#Z"6, 7;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0xABB1FFFFFFE6001AU;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x5867FF9A0797FFEAU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0x70E02D0B29EAFF01U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x9C590282328BB651U;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0xE26D1988810EA9A0U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xEC53755D84F84302U;" \
    "\n@gt\tsubc.u64    "#Z"6, "#Z"6, 7;" \
 \
    /* if z >= 2^386 then z -= 39m */ \
 \
    "\n\tsetp.gt.u64 gt, "#Z"6, 3;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0x55D8FFFFFFF3000DU;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0xAC33FFCD03CBFFF5U;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0xB870168594F57F80U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x4E2C81411945DB28U;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0x71368CC4408754D0U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xF629BAAEC27C2181U;" \
    "\n@gt\tsubc.u64    "#Z"6, "#Z"6, 3;" \
 \
    /* if z >= 2^385 then z -= 19m */ \
 \
    "\n\tsetp.gt.u64 gt, "#Z"6, 1;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0xCDECFFFFFFF9AAB1U;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x46C3FFE7293BFFFAU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0xA89FA1F24F2244AEU;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x74DA9ADE12E06434U;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0x930D7286FE9DD3FCU;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xEE145462447E1D73U;" \
    "\n@gt\tsubc.u64    "#Z"6, "#Z"6, 1;" \
 \
    /* if z >= 2^384 then z -= 9m */ \
 \
    "\n\tsetp.gt.u64 gt, "#Z"6, 0;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0x89F6FFFFFFFD0003U;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x140BFFF43BF3FFFDU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0xA0B767A8AC38A745U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x8831A7AC8FADA8BAU;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0xA3F8E5685DA91392U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xEA09A13C057F1B6CU;" \
    "\n@gt\tsubc.u64    "#Z"6, "#Z"6, 0;" \
 \
    /* if z >= 2^384 then z -= 9m */ \
 \
    "\n\tsetp.gt.u64 gt, "#Z"6, 0;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0x89F6FFFFFFFD0003U;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x140BFFF43BF3FFFDU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0xA0B767A8AC38A745U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x8831A7AC8FADA8BAU;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0xA3F8E5685DA91392U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xEA09A13C057F1B6CU;"

__forceinline__
__device__ void fp_x12(
    uint64_t &z0,
    uint64_t &z1,
    uint64_t &z2,
    uint64_t &z3,
    uint64_t &z4,
    uint64_t &z5,
    uint64_t x0,
    uint64_t x1,
    uint64_t x2,
    uint64_t x3,
    uint64_t x4,
    uint64_t x5) {

    asm volatile (
    "\n\t{"
    "\n\t.reg .u64 z<7>, x<6>;"
    "\n\t.reg .u64 t<6>;"
    "\n\t.reg .pred gt, nz;"

    "\n\tmov.u64 x0,  %6;"
    "\n\tmov.u64 x1,  %7;"
    "\n\tmov.u64 x2,  %8;"
    "\n\tmov.u64 x3,  %9;"
    "\n\tmov.u64 x4, %10;"
    "\n\tmov.u64 x5, %11;"

FP_X12(z, x, t)

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
    "l"(x0), "l"(x1), "l"(x2), "l"(x3), "l"(x4), "l"(x5)
    );
}

#endif
// vim: ts=4 et sw=4 si
