// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

/**
 * @brief Device macro for Z = X+Y with overflow check
 *
 * @param[out] Z destination register
 * @param[in ] X first operand
 * @param[in ] Y second operand
 * @return void
 *
 */
#define FR_ADD(Z, X, Y) \
    /* z = x + y */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#X"0, "#Y"0;" \
    "\n\taddc.u64.cc "#Z"1, "#X"1, "#Y"1;" \
    "\n\taddc.u64.cc "#Z"2, "#X"2, "#Y"2;" \
    "\n\taddc.u64.cc "#Z"3, "#X"3, "#Y"3;" \
    "\n\taddc.u32    t4,  0,   0;" \
 \
    /* if z >= 2^256 then z -= rmmu0 */ \
 \
    "\n\tsetp.ne.u32 nz, t4, 0;" \
    "\n@nz\tsub.u64.cc  "#Z"0, "#Z"0, 0xFFFFFFFE00000002U;" \
    "\n@nz\tsubc.u64.cc "#Z"1, "#Z"1, 0xA77B4805FFFCB7FDU;" \
    "\n@nz\tsubc.u64.cc "#Z"2, "#Z"2, 0x6673B0101343B00AU;" \
    "\n@nz\tsubc.u64.cc "#Z"3, "#Z"3, 0xE7DB4EA6533AFA90U;" \
    "\n@nz\tsubc.u32    t4, t4, 0;" \
 \
    /* if z >= 2^256 then z -= rmmu0 */ \
 \
    "\n\tsetp.ne.u32 nz, t4, 0;" \
    "\n@nz\tsub.u64.cc  "#Z"0, "#Z"0, 0xFFFFFFFE00000002U;" \
    "\n@nz\tsubc.u64.cc "#Z"1, "#Z"1, 0xA77B4805FFFCB7FDU;" \
    "\n@nz\tsubc.u64.cc "#Z"2, "#Z"2, 0x6673B0101343B00AU;" \
    "\n@nz\tsubc.u64.cc "#Z"3, "#Z"3, 0xE7DB4EA6533AFA90U;"

