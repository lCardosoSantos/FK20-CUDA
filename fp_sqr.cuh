#ifndef FP_SQR

// Needs Z0..Zb, q0..q7, r0..r6

#define FP_SQR(Z, X) \
\
    "\n\tmul.lo.u64     "#Z"5, "#X"0, "#X"5    ; mul.hi.u64     "#Z"6, "#X"0, "#X"5    ;" \
\
    "\n\tmul.lo.u64     "#Z"4, "#X"0, "#X"4    ; mad.hi.u64.cc  "#Z"5, "#X"0, "#X"4, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"1, "#X"5, "#Z"6; madc.hi.u64    "#Z"7, "#X"1, "#X"5,  0;" \
\
    "\n\tmul.lo.u64     "#Z"3, "#X"0, "#X"3    ; mad.hi.u64.cc  "#Z"4, "#X"0, "#X"3, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"1, "#X"4, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"1, "#X"4, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"2, "#X"5, "#Z"7; madc.hi.u64    "#Z"8, "#X"2, "#X"5,  0;" \
\
    "\n\tmul.lo.u64     "#Z"2, "#X"0, "#X"2    ; mad.hi.u64.cc  "#Z"3, "#X"0, "#X"2, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"1, "#X"3, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"1, "#X"3, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"2, "#X"4, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"2, "#X"4, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"3, "#X"5, "#Z"8; madc.hi.u64    "#Z"9, "#X"3, "#X"5,  0;" \
\
    "\n\tmul.lo.u64     "#Z"1, "#X"0, "#X"1    ; mad.hi.u64.cc  "#Z"2, "#X"0, "#X"1, "#Z"2;" \
    "\n\tmadc.lo.u64.cc "#Z"3, "#X"1, "#X"2, "#Z"3; madc.hi.u64.cc "#Z"4, "#X"1, "#X"2, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"2, "#X"3, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"2, "#X"3, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"3, "#X"4, "#Z"7; madc.hi.u64.cc "#Z"8, "#X"3, "#X"4, "#Z"8;" \
    "\n\tmadc.lo.u64.cc "#Z"9, "#X"4, "#X"5, "#Z"9; madc.hi.u64    "#Z"a, "#X"4, "#X"5,  0;" \
\
    "\n\tadd.u64.cc  "#Z"1, "#Z"1, "#Z"1;" \
    "\n\taddc.u64.cc "#Z"2, "#Z"2, "#Z"2;" \
    "\n\taddc.u64.cc "#Z"3, "#Z"3, "#Z"3;" \
    "\n\taddc.u64.cc "#Z"4, "#Z"4, "#Z"4;" \
    "\n\taddc.u64.cc "#Z"5, "#Z"5, "#Z"5;" \
    "\n\taddc.u64.cc "#Z"6, "#Z"6, "#Z"6;" \
    "\n\taddc.u64.cc "#Z"7, "#Z"7, "#Z"7;" \
    "\n\taddc.u64.cc "#Z"8, "#Z"8, "#Z"8;" \
    "\n\taddc.u64.cc "#Z"9, "#Z"9, "#Z"9;" \
    "\n\taddc.u64.cc "#Z"a, "#Z"a, "#Z"a;" \
    "\n\taddc.u64    "#Z"b,  0,  0;" \
\
    "\n\tmul.lo.u64     "#Z"0, "#X"0, "#X"0    ; mad.hi.u64.cc  "#Z"1, "#X"0, "#X"0, "#Z"1;" \
    "\n\tmadc.lo.u64.cc "#Z"2, "#X"1, "#X"1, "#Z"2; madc.hi.u64.cc "#Z"3, "#X"1, "#X"1, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"2, "#X"2, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"2, "#X"2, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"3, "#X"3, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"3, "#X"3, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"4, "#X"4, "#Z"8; madc.hi.u64.cc "#Z"9, "#X"4, "#X"4, "#Z"9;" \
    "\n\tmadc.lo.u64.cc "#Z"a, "#X"5, "#X"5, "#Z"a; madc.hi.u64    "#Z"b, "#X"5, "#X"5, "#Z"b;"

#endif
// vim: ts=4 et sw=4 si
