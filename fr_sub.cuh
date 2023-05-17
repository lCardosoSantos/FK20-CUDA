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
    "\n\tsetp.ne.u32 cp, t4, 0;" \
    "\n@cp\tadd.u64.cc  "#Z"0, "#Z"0, 0xFFFFFFFE00000002U;" \
    "\n@cp\taddc.u64.cc "#Z"1, "#Z"1, 0xA77B4805FFFCB7FDU;" \
    "\n@cp\taddc.u64.cc "#Z"2, "#Z"2, 0x6673B0101343B00AU;" \
    "\n@cp\taddc.u64.cc "#Z"3, "#Z"3, 0xE7DB4EA6533AFA90U;" \
    "\n@cp\taddc.u32    t4, t4, 0;" \
 \
    /* if z < 0 then z += rmmu0 */ \
 \
    "\n\tsetp.ne.u32 cp, t4, 0;" \
    "\n@cp\tadd.u64.cc  "#Z"0, "#Z"0, 0xFFFFFFFE00000002U;" \
    "\n@cp\taddc.u64.cc "#Z"1, "#Z"1, 0xA77B4805FFFCB7FDU;" \
    "\n@cp\taddc.u64.cc "#Z"2, "#Z"2, 0x6673B0101343B00AU;" \
    "\n@cp\taddc.u64.cc "#Z"3, "#Z"3, 0xE7DB4EA6533AFA90U;" \

