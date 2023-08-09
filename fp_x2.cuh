// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef FP_X2

/**
 * @brief PTX macro for multiplication by 2. Stores in Z.
 * 
 */
#define FP_X2(Z, X) \
    /* z = x + x */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#X"0, "#X"0;" \
    "\n\taddc.u64.cc "#Z"1, "#X"1, "#X"1;" \
    "\n\taddc.u64.cc "#Z"2, "#X"2, "#X"2;" \
    "\n\taddc.u64.cc "#Z"3, "#X"3, "#X"3;" \
    "\n\taddc.u64.cc "#Z"4, "#X"4, "#X"4;" \
    "\n\taddc.u64.cc "#Z"5, "#X"5, "#X"5;" \
    "\n\taddc.u32    z6,  0,   0;" \
 \
    /* gt = (z>>320) > (m>>320) */ \
    /* nz = (z>>384) > 0 */ \
 \
    "\n\tsetp.gt.u64 gt, z5, 0x1a0111ea397fe69aU;" \
    "\n\tsetp.ne.u32 nz, z6, 0;" \
 \
    /* If gt then subtract m */ \
 \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0xb9feffffffffaaabU;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x1eabfffeb153ffffU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0x6730d2a0f6b0f624U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x64774b84f38512bfU;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0x4b1ba7b6434bacd7U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0x1a0111ea397fe69aU;" \
 \
    /* If nz then subtract mmu0 (= 9m) */ \
 \
    "\n@nz\tsub.u64.cc  "#Z"0, "#Z"0, 0x89F6FFFFFFFD0003U;" \
    "\n@nz\tsubc.u64.cc "#Z"1, "#Z"1, 0x140BFFF43BF3FFFDU;" \
    "\n@nz\tsubc.u64.cc "#Z"2, "#Z"2, 0xA0B767A8AC38A745U;" \
    "\n@nz\tsubc.u64.cc "#Z"3, "#Z"3, 0x8831A7AC8FADA8BAU;" \
    "\n@nz\tsubc.u64.cc "#Z"4, "#Z"4, 0xA3F8E5685DA91392U;" \
    "\n@nz\tsubc.u64.cc "#Z"5, "#Z"5, 0xEA09A13C057F1B6CU;"

#endif
