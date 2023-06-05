#ifndef FP_X12

#define FP_X12(Z, X) \
    /* t = x + x */ \
 \
    "\n\tadd.u64.cc  t0, "#X"0, "#X"0;" \
    "\n\taddc.u64.cc t1, "#X"1, "#X"1;" \
    "\n\taddc.u64.cc t2, "#X"2, "#X"2;" \
    "\n\taddc.u64.cc t3, "#X"3, "#X"3;" \
    "\n\taddc.u64.cc t4, "#X"4, "#X"4;" \
    "\n\taddc.u64.cc t5, "#X"5, "#X"5;" \
    "\n\taddc.u32    z6,     0,     0;" \
 \
    /* z = x + t */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#X"0, t0;" \
    "\n\taddc.u64.cc "#Z"1, "#X"1, t1;" \
    "\n\taddc.u64.cc "#Z"2, "#X"2, t2;" \
    "\n\taddc.u64.cc "#Z"3, "#X"3, t3;" \
    "\n\taddc.u64.cc "#Z"4, "#X"4, t4;" \
    "\n\taddc.u64.cc "#Z"5, "#X"5, t5;" \
    "\n\taddc.u32       z6,     0, z6;" \
 \
    /* z = z + z */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#Z"0, "#Z"0;" \
    "\n\taddc.u64.cc "#Z"1, "#Z"1, "#Z"1;" \
    "\n\taddc.u64.cc "#Z"2, "#Z"2, "#Z"2;" \
    "\n\taddc.u64.cc "#Z"3, "#Z"3, "#Z"3;" \
    "\n\taddc.u64.cc "#Z"4, "#Z"4, "#Z"4;" \
    "\n\taddc.u64.cc "#Z"5, "#Z"5, "#Z"5;" \
    "\n\taddc.u32       z6,    z6,    z6;" \
 \
    /* z = z + z */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#Z"0, "#Z"0;" \
    "\n\taddc.u64.cc "#Z"1, "#Z"1, "#Z"1;" \
    "\n\taddc.u64.cc "#Z"2, "#Z"2, "#Z"2;" \
    "\n\taddc.u64.cc "#Z"3, "#Z"3, "#Z"3;" \
    "\n\taddc.u64.cc "#Z"4, "#Z"4, "#Z"4;" \
    "\n\taddc.u64.cc "#Z"5, "#Z"5, "#Z"5;" \
    "\n\taddc.u32       z6,    z6,    z6;" \
 \
    /* if z >= 2^387 then z -= 78m */ \
 \
    "\n\tsetp.gt.u32 gt, z6, 7;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0xABB1FFFFFFE6001AU;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x5867FF9A0797FFEAU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0x70E02D0B29EAFF01U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x9C590282328BB651U;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0xE26D1988810EA9A0U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xEC53755D84F84302U;" \
    "\n@gt\tsubc.u32       z6,    z6, 7;" \
 \
    /* if z >= 2^386 then z -= 39m */ \
 \
    "\n\tsetp.gt.u32 gt, z6, 3;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0x55D8FFFFFFF3000DU;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0xAC33FFCD03CBFFF5U;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0xB870168594F57F80U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x4E2C81411945DB28U;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0x71368CC4408754D0U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xF629BAAEC27C2181U;" \
    "\n@gt\tsubc.u32       z6,    z6, 3;" \
 \
    /* if z >= 2^385 then z -= 19m */ \
 \
    "\n\tsetp.gt.u32 gt, z6, 1;" \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0xCDECFFFFFFF9AAB1U;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x46C3FFE7293BFFFAU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0xA89FA1F24F2244AEU;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x74DA9ADE12E06434U;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0x930D7286FE9DD3FCU;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0xEE145462447E1D73U;" \
    "\n@gt\tsubc.u32       z6,    z6, 1;" \
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
