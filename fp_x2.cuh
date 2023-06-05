#ifndef FP_X2

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
    /* if z >= 2^384 then z -= 9m */ \
 \
    "\n\tsetp.gt.u32 gt, z6, 0;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0x89F6FFFFFFFD0003U;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x140BFFF43BF3FFFDU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0xA0B767A8AC38A745U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x8831A7AC8FADA8BAU;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0xA3F8E5685DA91392U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xEA09A13C057F1B6CU;" \
    "\n@gt\tsubc.u32       z6,    z6, 0;" \
 \
    /* if z >= 2^384 then z -= 9m */ \
 \
    "\n\tsetp.gt.u32 gt, z6, 0;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0x89F6FFFFFFFD0003U;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x140BFFF43BF3FFFDU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0xA0B767A8AC38A745U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x8831A7AC8FADA8BAU;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0xA3F8E5685DA91392U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xEA09A13C057F1B6CU;"

#endif
