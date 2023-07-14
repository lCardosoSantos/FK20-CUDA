// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"
#include "fk20_testvector.cuh"

int main() {
    FK20TestFFT();
    FK20TestPoly();
    //FK20TestFFTRand(NULL); 
    // TODO: Luan's note: This function here hasn't been updated in a while. 
    // Probably not worth the effort to debug it, since the scope of it is now 
    // covered by the 512 row tests.

    return 0;
}

// vim: ts=4 et sw=4 si