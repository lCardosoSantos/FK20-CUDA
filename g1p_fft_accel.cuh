// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef G1P_MUL_ROOT_CUH
#define G1P_MUL_ROOT_CUH

enum {
    dbl = 0,  // P0 ← P0 * 2
    add1,     // P0 ← P0 + P1
    sub1,     // P0 ← P0 - P1
    add3,     // P0 ← P0 + P3
    sub3,     // P0 ← P0 - P3
    st1,      // P1 ← P0
    st3,      // P3 ← P0
    sw3,      // P0,P3 ← P3,P0
    ldp,      // P0 ← P
    stp,      // P ← P0
    ldq,      // P0 ← Q
    stq,      // Q ← P0
    end       // Return.
};

#endif

// vim: ts=4 et sw=4 si
