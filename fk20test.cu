// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"
#include "fk20_testvector.cuh"
int main() {
    FK20TestPoly();
    FK20TestFFT();
    FK20TestPoly();
    FK20TestFFTRand(NULL);

    //Minimal case that causes (detectable) memory issues (failure of fk20_poly2h_fft_test).
    //fk20_poly2h_fft_test(1);
//
    //h2h_fft();          // Any of these two functions, or both
    //hext_fft2h();       // but not none
//
    //fk20_poly2h_fft_test(2);
    return 0;
}

// vim: ts=4 et sw=4 si
