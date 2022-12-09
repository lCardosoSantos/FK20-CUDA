/* Reads X0..X5 and Y0..Y5. Writes u0..ub */

#define FP_MUL(X, Y) \
    "\n\tmul.lo.u64     u1, "#X"0, "#Y"1    ; mul.hi.u64     u2, "#X"0, "#Y"1    ;" \
    "\n\tmul.lo.u64     u3, "#X"0, "#Y"3    ; mul.hi.u64     u4, "#X"0, "#Y"3    ;" \
    "\n\tmul.lo.u64     u5, "#X"0, "#Y"5    ; mul.hi.u64     u6, "#X"0, "#Y"5    ;" \
\
    "\n\tmul.lo.u64     u0, "#X"0, "#Y"0    ; mad.hi.u64.cc  u1, "#X"0, "#Y"0, u1;" \
    "\n\tmadc.lo.u64.cc u2, "#X"0, "#Y"2, u2; madc.hi.u64.cc u3, "#X"0, "#Y"2, u3;" \
    "\n\tmadc.lo.u64.cc u4, "#X"0, "#Y"4, u4; madc.hi.u64.cc u5, "#X"0, "#Y"4, u5;" \
    "\n\taddc.u64       u6,  0,  u6;" \
\
\
    "\n\tmad.lo.u64.cc  u2, "#X"1, "#Y"1, u2; madc.hi.u64.cc u3, "#X"1, "#Y"1, u3;" \
    "\n\tmadc.lo.u64.cc u4, "#X"1, "#Y"3, u4; madc.hi.u64.cc u5, "#X"1, "#Y"3, u5;" \
    "\n\tmadc.lo.u64.cc u6, "#X"1, "#Y"5, u6; madc.hi.u64    u7, "#X"1, "#Y"5,  0;" \
\
    "\n\tmad.lo.u64.cc  u1, "#X"1, "#Y"0, u1; madc.hi.u64.cc u2, "#X"1, "#Y"0, u2;" \
    "\n\tmadc.lo.u64.cc u3, "#X"1, "#Y"2, u3; madc.hi.u64.cc u4, "#X"1, "#Y"2, u4;" \
    "\n\tmadc.lo.u64.cc u5, "#X"1, "#Y"4, u5; madc.hi.u64.cc u6, "#X"1, "#Y"4, u6;" \
    "\n\taddc.u64       u7,  0,  u7;" \
\
\
    "\n\tmad.lo.u64.cc  u3, "#X"2, "#Y"1, u3; madc.hi.u64.cc u4, "#X"2, "#Y"1, u4;" \
    "\n\tmadc.lo.u64.cc u5, "#X"2, "#Y"3, u5; madc.hi.u64.cc u6, "#X"2, "#Y"3, u6;" \
    "\n\tmadc.lo.u64.cc u7, "#X"2, "#Y"5, u7; madc.hi.u64    u8, "#X"2, "#Y"5,  0;" \
\
    "\n\tmad.lo.u64.cc  u2, "#X"2, "#Y"0, u2; madc.hi.u64.cc u3, "#X"2, "#Y"0, u3;" \
    "\n\tmadc.lo.u64.cc u4, "#X"2, "#Y"2, u4; madc.hi.u64.cc u5, "#X"2, "#Y"2, u5;" \
    "\n\tmadc.lo.u64.cc u6, "#X"2, "#Y"4, u6; madc.hi.u64.cc u7, "#X"2, "#Y"4, u7;" \
    "\n\taddc.u64       u8,  0,  u8;" \
\
\
    "\n\tmad.lo.u64.cc  u4, "#X"3, "#Y"1, u4; madc.hi.u64.cc u5, "#X"3, "#Y"1, u5;" \
    "\n\tmadc.lo.u64.cc u6, "#X"3, "#Y"3, u6; madc.hi.u64.cc u7, "#X"3, "#Y"3, u7;" \
    "\n\tmadc.lo.u64.cc u8, "#X"3, "#Y"5, u8; madc.hi.u64    u9, "#X"3, "#Y"5,  0;" \
\
    "\n\tmad.lo.u64.cc  u3, "#X"3, "#Y"0, u3; madc.hi.u64.cc u4, "#X"3, "#Y"0, u4;" \
    "\n\tmadc.lo.u64.cc u5, "#X"3, "#Y"2, u5; madc.hi.u64.cc u6, "#X"3, "#Y"2, u6;" \
    "\n\tmadc.lo.u64.cc u7, "#X"3, "#Y"4, u7; madc.hi.u64.cc u8, "#X"3, "#Y"4, u8;" \
    "\n\taddc.u64       u9,  0,  u9;" \
\
\
    "\n\tmad.lo.u64.cc  u5, "#X"4, "#Y"1, u5; madc.hi.u64.cc u6, "#X"4, "#Y"1, u6;" \
    "\n\tmadc.lo.u64.cc u7, "#X"4, "#Y"3, u7; madc.hi.u64.cc u8, "#X"4, "#Y"3, u8;" \
    "\n\tmadc.lo.u64.cc u9, "#X"4, "#Y"5, u9; madc.hi.u64    ua, "#X"4, "#Y"5,  0;" \
\
    "\n\tmad.lo.u64.cc  u4, "#X"4, "#Y"0, u4; madc.hi.u64.cc u5, "#X"4, "#Y"0, u5;" \
    "\n\tmadc.lo.u64.cc u6, "#X"4, "#Y"2, u6; madc.hi.u64.cc u7, "#X"4, "#Y"2, u7;" \
    "\n\tmadc.lo.u64.cc u8, "#X"4, "#Y"4, u8; madc.hi.u64.cc u9, "#X"4, "#Y"4, u9;" \
    "\n\taddc.u64       ua,  0,  ua;" \
\
\
    "\n\tmad.lo.u64.cc  u6, "#X"5, "#Y"1, u6; madc.hi.u64.cc u7, "#X"5, "#Y"1, u7;" \
    "\n\tmadc.lo.u64.cc u8, "#X"5, "#Y"3, u8; madc.hi.u64.cc u9, "#X"5, "#Y"3, u9;" \
    "\n\tmadc.lo.u64.cc ua, "#X"5, "#Y"5, ua; madc.hi.u64    ub, "#X"5, "#Y"5,  0;" \
\
    "\n\tmad.lo.u64.cc  u5, "#X"5, "#Y"0, u5; madc.hi.u64.cc u6, "#X"5, "#Y"0, u6;" \
    "\n\tmadc.lo.u64.cc u7, "#X"5, "#Y"2, u7; madc.hi.u64.cc u8, "#X"5, "#Y"2, u8;" \
    "\n\tmadc.lo.u64.cc u9, "#X"5, "#Y"4, u9; madc.hi.u64.cc ua, "#X"5, "#Y"4, ua;" \
    "\n\taddc.u64       ub,  0,  ub;"

// vim: ts=4 et sw=4 si
