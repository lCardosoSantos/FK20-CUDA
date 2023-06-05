#ifndef FP_ADD

#define FP_ADD(Z, X, Y) \
    /* z = x + y */ \
 \
    "\n\tadd.u64.cc  "#Z"0, "#X"0, "#Y"0;" \
    "\n\taddc.u64.cc "#Z"1, "#X"1, "#Y"1;" \
    "\n\taddc.u64.cc "#Z"2, "#X"2, "#Y"2;" \
    "\n\taddc.u64.cc "#Z"3, "#X"3, "#Y"3;" \
    "\n\taddc.u64.cc "#Z"4, "#X"4, "#Y"4;" \
    "\n\taddc.u64.cc "#Z"5, "#X"5, "#Y"5;" \
    "\n\taddc.u32    z6,  0,   0;" \
 \
    "\n\tsetp.gt.u64 gt, z5, 0x1a0111ea397fe69aU;" \
    "\n\tsetp.ne.u32 ne, z6, 0;" \
 \
    /* If digit 5 of x+y > digit 5 of m then subtract m, */ \
    /* AND if x+y >= 2^384 then subtract mmu0 (= 9m) */ \
 \
    "\n@gt\tsub.u64.cc  "#Z"0, "#Z"0, 0xb9feffffffffaaabU;" \
    "\n@gt\tsubc.u64.cc "#Z"1, "#Z"1, 0x1eabfffeb153ffffU;" \
    "\n@gt\tsubc.u64.cc "#Z"2, "#Z"2, 0x6730d2a0f6b0f624U;" \
    "\n@gt\tsubc.u64.cc "#Z"3, "#Z"3, 0x64774b84f38512bfU;" \
    "\n@gt\tsubc.u64.cc "#Z"4, "#Z"4, 0x4b1ba7b6434bacd7U;" \
    "\n@gt\tsubc.u64.cc "#Z"5, "#Z"5, 0x1a0111ea397fe69aU;" \
 \
    "\n@ne\tsub.u64.cc  "#Z"0, "#Z"0, 0x89f6fffffffd0003U;" \
    "\n@ne\tsubc.u64.cc "#Z"1, "#Z"1, 0x140bfff43bf3fffdU;" \
    "\n@ne\tsubc.u64.cc "#Z"2, "#Z"2, 0xa0b767a8ac38a745U;" \
    "\n@ne\tsubc.u64.cc "#Z"3, "#Z"3, 0x8831a7ac8fada8baU;" \
    "\n@ne\tsubc.u64.cc "#Z"4, "#Z"4, 0xa3f8e5685da91392U;" \
    "\n@ne\tsubc.u64.cc "#Z"5, "#Z"5, 0xea09a13c057f1b6cU;"

#endif
