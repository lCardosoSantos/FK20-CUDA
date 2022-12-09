#define FP_X3(Z, X) \
    /* t = x + x */ \
 \
    "\n\tadd.u64.cc  t0, "#X"0, "#X"0;" \
    "\n\taddc.u64.cc t1, "#X"1, "#X"1;" \
    "\n\taddc.u64.cc t2, "#X"2, "#X"2;" \
    "\n\taddc.u64.cc t3, "#X"3, "#X"3;" \
    "\n\taddc.u64.cc t4, "#X"4, "#X"4;" \
    "\n\taddc.u64.cc t5, "#X"5, "#X"5;" \
    "\n\taddc.u32    t6,  0,   0;" \
 \
    /* if t >= 2^384 then t -= mmu0 */ \
 \
    "\n\tsetp.ge.u32 cp, t6, 1;" \
    "\n@cp\tsub.u64.cc  t0, t0, 0x89f6fffffffd0003U;" \
    "\n@cp\tsubc.u64.cc t1, t1, 0x140bfff43bf3fffdU;" \
    "\n@cp\tsubc.u64.cc t2, t2, 0xa0b767a8ac38a745U;" \
    "\n@cp\tsubc.u64.cc t3, t3, 0x8831a7ac8fada8baU;" \
    "\n@cp\tsubc.u64.cc t4, t4, 0xa3f8e5685da91392U;" \
    "\n@cp\tsubc.u64.cc t5, t5, 0xea09a13c057f1b6cU;" \
    "\n@cp\tsubc.u32    t6, t6, 0;" \
 \
    /* z = x + t */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#X"0, t0;" \
    "\n\taddc.u64.cc "#Z"1, "#X"1, t1;" \
    "\n\taddc.u64.cc "#Z"2, "#X"2, t2;" \
    "\n\taddc.u64.cc "#Z"3, "#X"3, t3;" \
    "\n\taddc.u64.cc "#Z"4, "#X"4, t4;" \
    "\n\taddc.u64.cc "#Z"5, "#X"5, t5;" \
    "\n\taddc.u32    t6, t6,   0;" \
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

