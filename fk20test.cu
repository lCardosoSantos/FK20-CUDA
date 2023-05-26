// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fk20test.cuh"

int main() {

    FK20TestPoly();
    FK20TestFFT();
    FK20TestFFTRand(NULL);


    return 0;
}

// vim: ts=4 et sw=4 si
