#define FP_ADD(Z, X, Y) \
    /* z = x + y */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#X"0, "#Y"0;" \
    "\n\taddc.u64.cc "#Z"1, "#X"1, "#Y"1;" \
    "\n\taddc.u64.cc "#Z"2, "#X"2, "#Y"2;" \
    "\n\taddc.u64.cc "#Z"3, "#X"3, "#Y"3;" \
    "\n\taddc.u64.cc "#Z"4, "#X"4, "#Y"4;" \
    "\n\taddc.u64.cc "#Z"5, "#X"5, "#Y"5;" \
    "\n\taddc.u32    t6,  0,   0;" \
 \
    /* if z >= 2^384 then z -= mmu0 */ \
 \
    "\n\tsetp.ge.u32 cp, t6, 1;" \
    "\n@cp\tsub.u64.cc  "#Z"0, "#Z"0, 0x89f6fffffffd0003U;" \
    "\n@cp\tsubc.u64.cc "#Z"1, "#Z"1, 0x140bfff43bf3fffdU;" \
    "\n@cp\tsubc.u64.cc "#Z"2, "#Z"2, 0xa0b767a8ac38a745U;" \
    "\n@cp\tsubc.u64.cc "#Z"3, "#Z"3, 0x8831a7ac8fada8baU;" \
    "\n@cp\tsubc.u64.cc "#Z"4, "#Z"4, 0xa3f8e5685da91392U;" \
    "\n@cp\tsubc.u64.cc "#Z"5, "#Z"5, 0xea09a13c057f1b6cU;" \
    "\n@cp\tsubc.u32    t6, t6, 0;" \
 \
    /* if z >= 2^384 then z -= mmu0 */ \
 \
    "\n\tsetp.ge.u32 cp, t6, 1;" \
    "\n@cp\tsub.u64.cc  "#Z"0, "#Z"0, 0x89f6fffffffd0003U;" \
    "\n@cp\tsubc.u64.cc "#Z"1, "#Z"1, 0x140bfff43bf3fffdU;" \
    "\n@cp\tsubc.u64.cc "#Z"2, "#Z"2, 0xa0b767a8ac38a745U;" \
    "\n@cp\tsubc.u64.cc "#Z"3, "#Z"3, 0x8831a7ac8fada8baU;" \
    "\n@cp\tsubc.u64.cc "#Z"4, "#Z"4, 0xa3f8e5685da91392U;" \
    "\n@cp\tsubc.u64.cc "#Z"5, "#Z"5, 0xea09a13c057f1b6cU;"

