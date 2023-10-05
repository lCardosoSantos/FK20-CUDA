// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

/**
 * @brief Macro for Z=X-Y. Consider that X is in registers X0..X3 and Y in Y0..Y3.
 * Z and X can overlap.
 *
 */
#define FR_SUB(Z, X, Y) \
    /* z = x - y */ \
 \
    "\n\tsub.u64.cc  "#Z"0, "#X"0, "#Y"0;" \
    "\n\tsubc.u64.cc "#Z"1, "#X"1, "#Y"1;" \
    "\n\tsubc.u64.cc "#Z"2, "#X"2, "#Y"2;" \
    "\n\tsubc.u64.cc "#Z"3, "#X"3, "#Y"3;" \
    "\n\tsubc.u32    t4,  0,   0;" \
 \
    /* if z < 0 then z += rmmu0 */ \
 \
    "\n\tsetp.ne.u32 nz, t4, 0;" \
    "\n@nz\tadd.u64.cc  "#Z"0, "#Z"0, 0xFFFFFFFE00000002U;" \
    "\n@nz\taddc.u64.cc "#Z"1, "#Z"1, 0xA77B4805FFFCB7FDU;" \
    "\n@nz\taddc.u64.cc "#Z"2, "#Z"2, 0x6673B0101343B00AU;" \
    "\n@nz\taddc.u64.cc "#Z"3, "#Z"3, 0xE7DB4EA6533AFA90U;" \
    "\n@nz\taddc.u32    t4, t4, 0;" \
 \
    /* if z < 0 then z += rmmu0 */ \
 \
    "\n\tsetp.ne.u32 nz, t4, 0;" \
    "\n@nz\tadd.u64.cc  "#Z"0, "#Z"0, 0xFFFFFFFE00000002U;" \
    "\n@nz\taddc.u64.cc "#Z"1, "#Z"1, 0xA77B4805FFFCB7FDU;" \
    "\n@nz\taddc.u64.cc "#Z"2, "#Z"2, 0x6673B0101343B00AU;" \
    "\n@nz\taddc.u64.cc "#Z"3, "#Z"3, 0xE7DB4EA6533AFA90U;" \

