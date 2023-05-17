// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef FK20_TEST_CUH
#define FK20_TEST_CUH

// Shared memory sizes

const size_t g1p_sharedmem = 512*3*6*8; // 512 points * 3 residues/point * 6 words/residue * 8 bytes/word = 72 KiB
const size_t fr_sharedmem = 512*4*8; // 512 residues * 4 words/residue * 8 bytes/word = 16 KiB

// Tests

void FK20TestFFT();
void FK20TestPoly();

#endif // FK20_TEST_CUH

// vim: ts=4 et sw=4 si
