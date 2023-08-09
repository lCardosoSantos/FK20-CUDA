#ifndef FP_MUL

/**
 * @brief PTX macro for multiplication of two residues mod p
 * Reads X0..X5 and Y0..Y5. Writes Z0..Zb
 * 
 */
#define FP_MUL(Z, X, Y) \
    "\n\tmul.lo.u64     "#Z"1, "#X"0, "#Y"1       ; mul.hi.u64     "#Z"2, "#X"0, "#Y"1       ;" \
    "\n\tmul.lo.u64     "#Z"3, "#X"0, "#Y"3       ; mul.hi.u64     "#Z"4, "#X"0, "#Y"3       ;" \
    "\n\tmul.lo.u64     "#Z"5, "#X"0, "#Y"5       ; mul.hi.u64     "#Z"6, "#X"0, "#Y"5       ;" \
\
    "\n\tmul.lo.u64     "#Z"0, "#X"0, "#Y"0       ; mad.hi.u64.cc  "#Z"1, "#X"0, "#Y"0, "#Z"1;" \
    "\n\tmadc.lo.u64.cc "#Z"2, "#X"0, "#Y"2, "#Z"2; madc.hi.u64.cc "#Z"3, "#X"0, "#Y"2, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"0, "#Y"4, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"0, "#Y"4, "#Z"5;" \
    "\n\taddc.u64       "#Z"6,     0, "#Z"6;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"2, "#X"1, "#Y"1, "#Z"2; madc.hi.u64.cc "#Z"3, "#X"1, "#Y"1, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"1, "#Y"3, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"1, "#Y"3, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"1, "#Y"5, "#Z"6; madc.hi.u64    "#Z"7, "#X"1, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"1, "#X"1, "#Y"0, "#Z"1; madc.hi.u64.cc "#Z"2, "#X"1, "#Y"0, "#Z"2;" \
    "\n\tmadc.lo.u64.cc "#Z"3, "#X"1, "#Y"2, "#Z"3; madc.hi.u64.cc "#Z"4, "#X"1, "#Y"2, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"1, "#Y"4, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"1, "#Y"4, "#Z"6;" \
    "\n\taddc.u64       "#Z"7,     0, "#Z"7;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"3, "#X"2, "#Y"1, "#Z"3; madc.hi.u64.cc "#Z"4, "#X"2, "#Y"1, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"2, "#Y"3, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"2, "#Y"3, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"2, "#Y"5, "#Z"7; madc.hi.u64    "#Z"8, "#X"2, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"2, "#X"2, "#Y"0, "#Z"2; madc.hi.u64.cc "#Z"3, "#X"2, "#Y"0, "#Z"3;" \
    "\n\tmadc.lo.u64.cc "#Z"4, "#X"2, "#Y"2, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"2, "#Y"2, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"2, "#Y"4, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"2, "#Y"4, "#Z"7;" \
    "\n\taddc.u64       "#Z"8,     0, "#Z"8;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"4, "#X"3, "#Y"1, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"3, "#Y"1, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"3, "#Y"3, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"3, "#Y"3, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"3, "#Y"5, "#Z"8; madc.hi.u64    "#Z"9, "#X"3, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"3, "#X"3, "#Y"0, "#Z"3; madc.hi.u64.cc "#Z"4, "#X"3, "#Y"0, "#Z"4;" \
    "\n\tmadc.lo.u64.cc "#Z"5, "#X"3, "#Y"2, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"3, "#Y"2, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"3, "#Y"4, "#Z"7; madc.hi.u64.cc "#Z"8, "#X"3, "#Y"4, "#Z"8;" \
    "\n\taddc.u64       "#Z"9,     0, "#Z"9;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"5, "#X"4, "#Y"1, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"4, "#Y"1, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"4, "#Y"3, "#Z"7; madc.hi.u64.cc "#Z"8, "#X"4, "#Y"3, "#Z"8;" \
    "\n\tmadc.lo.u64.cc "#Z"9, "#X"4, "#Y"5, "#Z"9; madc.hi.u64    "#Z"a, "#X"4, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"4, "#X"4, "#Y"0, "#Z"4; madc.hi.u64.cc "#Z"5, "#X"4, "#Y"0, "#Z"5;" \
    "\n\tmadc.lo.u64.cc "#Z"6, "#X"4, "#Y"2, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"4, "#Y"2, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"4, "#Y"4, "#Z"8; madc.hi.u64.cc "#Z"9, "#X"4, "#Y"4, "#Z"9;" \
    "\n\taddc.u64       "#Z"a,     0, "#Z"a;" \
\
\
    "\n\tmad.lo.u64.cc  "#Z"6, "#X"5, "#Y"1, "#Z"6; madc.hi.u64.cc "#Z"7, "#X"5, "#Y"1, "#Z"7;" \
    "\n\tmadc.lo.u64.cc "#Z"8, "#X"5, "#Y"3, "#Z"8; madc.hi.u64.cc "#Z"9, "#X"5, "#Y"3, "#Z"9;" \
    "\n\tmadc.lo.u64.cc "#Z"a, "#X"5, "#Y"5, "#Z"a; madc.hi.u64    "#Z"b, "#X"5, "#Y"5,     0;" \
\
    "\n\tmad.lo.u64.cc  "#Z"5, "#X"5, "#Y"0, "#Z"5; madc.hi.u64.cc "#Z"6, "#X"5, "#Y"0, "#Z"6;" \
    "\n\tmadc.lo.u64.cc "#Z"7, "#X"5, "#Y"2, "#Z"7; madc.hi.u64.cc "#Z"8, "#X"5, "#Y"2, "#Z"8;" \
    "\n\tmadc.lo.u64.cc "#Z"9, "#X"5, "#Y"4, "#Z"9; madc.hi.u64.cc "#Z"a, "#X"5, "#Y"4, "#Z"a;" \
    "\n\taddc.u64       "#Z"b,     0, "#Z"b;"

#endif
// vim: ts=4 et sw=4 si
