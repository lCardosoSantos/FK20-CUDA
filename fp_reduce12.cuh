/* Reads u0..ub. Writes q0..q7, r0..r6 and u0..u5 */

#define FP_REDUCE12() \
\
    /* q2 = q1 * mu; q3 = q2 / 2^448 */ \
\
    /* mu0 */ \
\
    "\n\tmul.hi.u64     q0, 0x13E207F56591BA2EU, ua;" \
\
    "\n\tmad.lo.u64.cc  q0, 0x13E207F56591BA2EU, ub, q0;" \
    "\n\tmadc.hi.u64    q1, 0x13E207F56591BA2EU, ub,  0;" \
\
    /* mu1 */ \
\
    "\n\tmad.hi.u64.cc  q0, 0x997167A058F1C07BU, u9, q0;" \
    "\n\tmadc.lo.u64.cc q1, 0x997167A058F1C07BU, ub, q1;" \
    "\n\tmadc.hi.u64    q2, 0x997167A058F1C07BU, ub,  0;" \
\
    "\n\tmad.lo.u64.cc  q0, 0x997167A058F1C07BU, ua, q0;" \
    "\n\tmadc.hi.u64.cc q1, 0x997167A058F1C07BU, ua, q1;" \
    "\n\taddc.u64       q2, q2, 0;" \
\
    /* mu2 */ \
\
    "\n\tmad.lo.u64.cc  q0, 0xDF4771E0286779D3U, u9, q0;" \
    "\n\tmadc.hi.u64.cc q1, 0xDF4771E0286779D3U, u9, q1;" \
    "\n\tmadc.lo.u64.cc q2, 0xDF4771E0286779D3U, ub, q2;" \
    "\n\tmadc.hi.u64    q3, 0xDF4771E0286779D3U, ub,  0;" \
\
    "\n\tmad.hi.u64.cc  q0, 0xDF4771E0286779D3U, u8, q0;" \
    "\n\tmadc.lo.u64.cc q1, 0xDF4771E0286779D3U, ua, q1;" \
    "\n\tmadc.hi.u64.cc q2, 0xDF4771E0286779D3U, ua, q2;" \
    "\n\taddc.u64       q3, q3, 0;" \
\
    /* mu3 */ \
\
    "\n\tmad.hi.u64.cc  q0, 0x1B82741FF6A0A94BU, u7, q0;" \
    "\n\tmadc.lo.u64.cc q1, 0x1B82741FF6A0A94BU, u9, q1;" \
    "\n\tmadc.hi.u64.cc q2, 0x1B82741FF6A0A94BU, u9, q2;" \
    "\n\tmadc.lo.u64.cc q3, 0x1B82741FF6A0A94BU, ub, q3;" \
    "\n\tmadc.hi.u64    q4, 0x1B82741FF6A0A94BU, ub,  0;" \
\
    "\n\tmad.lo.u64.cc  q0, 0x1B82741FF6A0A94BU, u8, q0;" \
    "\n\tmadc.hi.u64.cc q1, 0x1B82741FF6A0A94BU, u8, q1;" \
    "\n\tmadc.lo.u64.cc q2, 0x1B82741FF6A0A94BU, ua, q2;" \
    "\n\tmadc.hi.u64.cc q3, 0x1B82741FF6A0A94BU, ua, q3;" \
    "\n\taddc.u64       q4, q4, 0;" \
\
    /* mu4 */ \
\
    "\n\tmad.lo.u64.cc  q0, 0x28101B0CC7A6BA29U, u7, q0;" \
    "\n\tmadc.hi.u64.cc q1, 0x28101B0CC7A6BA29U, u7, q1;" \
    "\n\tmadc.lo.u64.cc q2, 0x28101B0CC7A6BA29U, u9, q2;" \
    "\n\tmadc.hi.u64.cc q3, 0x28101B0CC7A6BA29U, u9, q3;" \
    "\n\tmadc.lo.u64.cc q4, 0x28101B0CC7A6BA29U, ub, q4;" \
    "\n\tmadc.hi.u64    q5, 0x28101B0CC7A6BA29U, ub,  0;" \
\
    "\n\tmad.hi.u64.cc  q0, 0x28101B0CC7A6BA29U, u6, q0;" \
    "\n\tmadc.lo.u64.cc q1, 0x28101B0CC7A6BA29U, u8, q1;" \
    "\n\tmadc.hi.u64.cc q2, 0x28101B0CC7A6BA29U, u8, q2;" \
    "\n\tmadc.lo.u64.cc q3, 0x28101B0CC7A6BA29U, ua, q3;" \
    "\n\tmadc.hi.u64.cc q4, 0x28101B0CC7A6BA29U, ua, q4;" \
    "\n\taddc.u64       q5, q5, 0;" \
\
    /* mu5 */ \
\
    "\n\tmad.hi.u64.cc  q0, 0xD835D2F3CC9E45CEU, u5, q0;" \
    "\n\tmadc.lo.u64.cc q1, 0xD835D2F3CC9E45CEU, u7, q1;" \
    "\n\tmadc.hi.u64.cc q2, 0xD835D2F3CC9E45CEU, u7, q2;" \
    "\n\tmadc.lo.u64.cc q3, 0xD835D2F3CC9E45CEU, u9, q3;" \
    "\n\tmadc.hi.u64.cc q4, 0xD835D2F3CC9E45CEU, u9, q4;" \
    "\n\tmadc.lo.u64.cc q5, 0xD835D2F3CC9E45CEU, ub, q5;" \
    "\n\tmadc.hi.u64    q6, 0xD835D2F3CC9E45CEU, ub,  0;" \
\
    "\n\tmad.lo.u64.cc  q0, 0xD835D2F3CC9E45CEU, u6, q0;" \
    "\n\tmadc.hi.u64.cc q1, 0xD835D2F3CC9E45CEU, u6, q1;" \
    "\n\tmadc.lo.u64.cc q2, 0xD835D2F3CC9E45CEU, u8, q2;" \
    "\n\tmadc.hi.u64.cc q3, 0xD835D2F3CC9E45CEU, u8, q3;" \
    "\n\tmadc.lo.u64.cc q4, 0xD835D2F3CC9E45CEU, ua, q4;" \
    "\n\tmadc.hi.u64.cc q5, 0xD835D2F3CC9E45CEU, ua, q5;" \
    "\n\taddc.u64       q6, q6, 0;" \
\
    /* mu6 */ \
\
    "\n\tmad.lo.u64.cc  q0, 0x0000000000000009U, u5, q0;" \
    "\n\tmadc.hi.u64.cc q1, 0x0000000000000009U, u5, q1;" \
    "\n\tmadc.lo.u64.cc q2, 0x0000000000000009U, u7, q2;" \
    "\n\tmadc.hi.u64.cc q3, 0x0000000000000009U, u7, q3;" \
    "\n\tmadc.lo.u64.cc q4, 0x0000000000000009U, u9, q4;" \
    "\n\tmadc.hi.u64.cc q5, 0x0000000000000009U, u9, q5;" \
    "\n\tmadc.lo.u64.cc q6, 0x0000000000000009U, ub, q6;" \
    "\n\tmadc.hi.u64    q7, 0x0000000000000009U, ub,  0;" \
\
    "\n\tmad.hi.u64.cc  q0, 0x0000000000000009U, u4, q0;" \
    "\n\tmadc.lo.u64.cc q1, 0x0000000000000009U, u6, q1;" \
    "\n\tmadc.hi.u64.cc q2, 0x0000000000000009U, u6, q2;" \
    "\n\tmadc.lo.u64.cc q3, 0x0000000000000009U, u8, q3;" \
    "\n\tmadc.hi.u64.cc q4, 0x0000000000000009U, u8, q4;" \
    "\n\tmadc.lo.u64.cc q5, 0x0000000000000009U, ua, q5;" \
    "\n\tmadc.hi.u64.cc q6, 0x0000000000000009U, ua, q6;" \
    "\n\taddc.u64       q7, q7, 0;" \
\
    /* r2 = q3 * m mod 2^448 */ \
    /*  u contains z^2 */ \
    /*  q contains q3 */ \
    /*  produces r2 in r */ \
\
    /* m5 */ \
\
    "\n\tmul.lo.u64     r5, 0x1A0111EA397FE69AU, q1    ;" \
    "\n\tmul.hi.u64     r6, 0x1A0111EA397FE69AU, q1    ;" \
    "\n\tmad.lo.u64     r6, 0x1A0111EA397FE69AU, q2, r6;" \
\
    /* m4 */ \
\
    "\n\tmul.lo.u64     r4, 0x4B1BA7B6434BACD7U, q1    ;" \
    "\n\tmad.hi.u64.cc  r5, 0x4B1BA7B6434BACD7U, q1, r5;" \
    "\n\tmadc.lo.u64    r6, 0x4B1BA7B6434BACD7U, q3, r6;" \
\
    "\n\tmad.lo.u64.cc  r5, 0x4B1BA7B6434BACD7U, q2, r5;" \
    "\n\tmadc.hi.u64    r6, 0x4B1BA7B6434BACD7U, q2, r6;" \
\
    /* m3 */ \
\
    "\n\tmul.lo.u64     r3, 0x64774B84F38512BFU, q1    ;" \
    "\n\tmad.hi.u64.cc  r4, 0x64774B84F38512BFU, q1, r4;" \
    "\n\tmadc.lo.u64.cc r5, 0x64774B84F38512BFU, q3, r5;" \
    "\n\tmadc.hi.u64    r6, 0x64774B84F38512BFU, q3, r6;" \
\
    "\n\tmad.lo.u64.cc  r4, 0x64774B84F38512BFU, q2, r4;" \
    "\n\tmadc.hi.u64.cc r5, 0x64774B84F38512BFU, q2, r5;" \
    "\n\tmadc.lo.u64    r6, 0x64774B84F38512BFU, q4, r6;" \
\
    /* m2 */ \
\
    "\n\tmul.lo.u64     r2, 0x6730D2A0F6B0F624U, q1    ;" \
    "\n\tmad.hi.u64.cc  r3, 0x6730D2A0F6B0F624U, q1, r3;" \
    "\n\tmadc.lo.u64.cc r4, 0x6730D2A0F6B0F624U, q3, r4;" \
    "\n\tmadc.hi.u64.cc r5, 0x6730D2A0F6B0F624U, q3, r5;" \
    "\n\tmadc.lo.u64    r6, 0x6730D2A0F6B0F624U, q5, r6;" \
\
    "\n\tmad.lo.u64.cc  r3, 0x6730D2A0F6B0F624U, q2, r3;" \
    "\n\tmadc.hi.u64.cc r4, 0x6730D2A0F6B0F624U, q2, r4;" \
    "\n\tmadc.lo.u64.cc r5, 0x6730D2A0F6B0F624U, q4, r5;" \
    "\n\tmadc.hi.u64    r6, 0x6730D2A0F6B0F624U, q4, r6;" \
\
    /* m1 */ \
\
    "\n\tmul.lo.u64     r1, 0x1EABFFFEB153FFFFU, q1    ;" \
    "\n\tmad.hi.u64.cc  r2, 0x1EABFFFEB153FFFFU, q1, r2;" \
    "\n\tmadc.lo.u64.cc r3, 0x1EABFFFEB153FFFFU, q3, r3;" \
    "\n\tmadc.hi.u64.cc r4, 0x1EABFFFEB153FFFFU, q3, r4;" \
    "\n\tmadc.lo.u64.cc r5, 0x1EABFFFEB153FFFFU, q5, r5;" \
    "\n\tmadc.hi.u64    r6, 0x1EABFFFEB153FFFFU, q5, r6;" \
\
    "\n\tmad.lo.u64.cc  r2, 0x1EABFFFEB153FFFFU, q2, r2;" \
    "\n\tmadc.hi.u64.cc r3, 0x1EABFFFEB153FFFFU, q2, r3;" \
    "\n\tmadc.lo.u64.cc r4, 0x1EABFFFEB153FFFFU, q4, r4;" \
    "\n\tmadc.hi.u64.cc r5, 0x1EABFFFEB153FFFFU, q4, r5;" \
    "\n\tmadc.lo.u64    r6, 0x1EABFFFEB153FFFFU, q6, r6;" \
\
    /* m0 */ \
\
    "\n\tmul.lo.u64     r0, 0xB9FEFFFFFFFFAAABU, q1    ;" \
    "\n\tmad.hi.u64.cc  r1, 0xB9FEFFFFFFFFAAABU, q1, r1;" \
    "\n\tmadc.lo.u64.cc r2, 0xB9FEFFFFFFFFAAABU, q3, r2;" \
    "\n\tmadc.hi.u64.cc r3, 0xB9FEFFFFFFFFAAABU, q3, r3;" \
    "\n\tmadc.lo.u64.cc r4, 0xB9FEFFFFFFFFAAABU, q5, r4;" \
    "\n\tmadc.hi.u64.cc r5, 0xB9FEFFFFFFFFAAABU, q5, r5;" \
    "\n\tmadc.lo.u64    r6, 0xB9FEFFFFFFFFAAABU, q7, r6;" \
\
    "\n\tmad.lo.u64.cc  r1, 0xB9FEFFFFFFFFAAABU, q2, r1;" \
    "\n\tmadc.hi.u64.cc r2, 0xB9FEFFFFFFFFAAABU, q2, r2;" \
    "\n\tmadc.lo.u64.cc r3, 0xB9FEFFFFFFFFAAABU, q4, r3;" \
    "\n\tmadc.hi.u64.cc r4, 0xB9FEFFFFFFFFAAABU, q4, r4;" \
    "\n\tmadc.lo.u64.cc r5, 0xB9FEFFFFFFFFAAABU, q6, r5;" \
    "\n\tmadc.hi.u64    r6, 0xB9FEFFFFFFFFAAABU, q6, r6;" \
\
    /* r = r1 - r2 */ \
    /*  r1 is in u */ \
    /*  r2 is in r */ \
\
    /* z = r1 - r2 */ \
\
    "\n\tsub.u64.cc  u0, u0, r0;" \
    "\n\tsubc.u64.cc u1, u1, r1;" \
    "\n\tsubc.u64.cc u2, u2, r2;" \
    "\n\tsubc.u64.cc u3, u3, r3;" \
    "\n\tsubc.u64.cc u4, u4, r4;" \
    "\n\tsubc.u64    u5, u5, r5;"

// vim: ts=4 et sw=4 si
