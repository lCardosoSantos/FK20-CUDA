// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "g1.cuh"
#include "fk20test.cuh"

void init() {
}

void FK20VerifyKAT();

//__managed__ testval_t testval[TESTVALS];

int main() {
    init();

    FK20TimeKAT();
    FK20VerifyKAT();

    return 0;
}

// vim: ts=4 et sw=4 si
