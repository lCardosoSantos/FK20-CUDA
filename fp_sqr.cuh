// Needs u0..ub, q0..q7, r0..r6

#define FP_SQR(X) \
\
    "\n\tmul.lo.u64     u5, "#X"0, "#X"5    ; mul.hi.u64     u6, "#X"0, "#X"5    ;" \
\
    "\n\tmul.lo.u64     u4, "#X"0, "#X"4    ; mad.hi.u64.cc  u5, "#X"0, "#X"4, u5;" \
    "\n\tmadc.lo.u64.cc u6, "#X"1, "#X"5, u6; madc.hi.u64    u7, "#X"1, "#X"5,  0;" \
\
    "\n\tmul.lo.u64     u3, "#X"0, "#X"3    ; mad.hi.u64.cc  u4, "#X"0, "#X"3, u4;" \
    "\n\tmadc.lo.u64.cc u5, "#X"1, "#X"4, u5; madc.hi.u64.cc u6, "#X"1, "#X"4, u6;" \
    "\n\tmadc.lo.u64.cc u7, "#X"2, "#X"5, u7; madc.hi.u64    u8, "#X"2, "#X"5,  0;" \
\
    "\n\tmul.lo.u64     u2, "#X"0, "#X"2    ; mad.hi.u64.cc  u3, "#X"0, "#X"2, u3;" \
    "\n\tmadc.lo.u64.cc u4, "#X"1, "#X"3, u4; madc.hi.u64.cc u5, "#X"1, "#X"3, u5;" \
    "\n\tmadc.lo.u64.cc u6, "#X"2, "#X"4, u6; madc.hi.u64.cc u7, "#X"2, "#X"4, u7;" \
    "\n\tmadc.lo.u64.cc u8, "#X"3, "#X"5, u8; madc.hi.u64    u9, "#X"3, "#X"5,  0;" \
\
    "\n\tmul.lo.u64     u1, "#X"0, "#X"1    ; mad.hi.u64.cc  u2, "#X"0, "#X"1, u2;" \
    "\n\tmadc.lo.u64.cc u3, "#X"1, "#X"2, u3; madc.hi.u64.cc u4, "#X"1, "#X"2, u4;" \
    "\n\tmadc.lo.u64.cc u5, "#X"2, "#X"3, u5; madc.hi.u64.cc u6, "#X"2, "#X"3, u6;" \
    "\n\tmadc.lo.u64.cc u7, "#X"3, "#X"4, u7; madc.hi.u64.cc u8, "#X"3, "#X"4, u8;" \
    "\n\tmadc.lo.u64.cc u9, "#X"4, "#X"5, u9; madc.hi.u64    ua, "#X"4, "#X"5,  0;" \
\
    "\n\tadd.u64.cc  u1, u1, u1;" \
    "\n\taddc.u64.cc u2, u2, u2;" \
    "\n\taddc.u64.cc u3, u3, u3;" \
    "\n\taddc.u64.cc u4, u4, u4;" \
    "\n\taddc.u64.cc u5, u5, u5;" \
    "\n\taddc.u64.cc u6, u6, u6;" \
    "\n\taddc.u64.cc u7, u7, u7;" \
    "\n\taddc.u64.cc u8, u8, u8;" \
    "\n\taddc.u64.cc u9, u9, u9;" \
    "\n\taddc.u64.cc ua, ua, ua;" \
    "\n\taddc.u64    ub,  0,  0;" \
\
    "\n\tmul.lo.u64     u0, "#X"0, "#X"0    ; mad.hi.u64.cc  u1, "#X"0, "#X"0, u1;" \
    "\n\tmadc.lo.u64.cc u2, "#X"1, "#X"1, u2; madc.hi.u64.cc u3, "#X"1, "#X"1, u3;" \
    "\n\tmadc.lo.u64.cc u4, "#X"2, "#X"2, u4; madc.hi.u64.cc u5, "#X"2, "#X"2, u5;" \
    "\n\tmadc.lo.u64.cc u6, "#X"3, "#X"3, u6; madc.hi.u64.cc u7, "#X"3, "#X"3, u7;" \
    "\n\tmadc.lo.u64.cc u8, "#X"4, "#X"4, u8; madc.hi.u64.cc u9, "#X"4, "#X"4, u9;" \
    "\n\tmadc.lo.u64.cc ua, "#X"5, "#X"5, ua; madc.hi.u64    ub, "#X"5, "#X"5, ub;"

// vim: ts=4 et sw=4 si
