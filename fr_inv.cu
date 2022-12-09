// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"
#include <stdio.h>

// Raise the argument to the power r-2.
// This avoids control flow divergence.
// 258 squarings, 55 multiplications.
__device__ void fr_inv(uint64_t *z) {

    // r-2 = 52435875175126190479447740508185965837690552500527637822603658699938581184511
    // = 111001111101101101001110101001100101001100111010111\
         110101001000001100110011100111011000000010000000100\
         110100001110110000000010101010011101111011010010000\
         000010111111111111111001011011111111101111111111111\
         111111111111111111011111111111111111111111111111111

    __shared__ uint64_t x1[4], x3[4], x5[4], x7[4];

    fr_cpy(x1, z);  // 1
    fr_sqr(z);      // 10

    fr_cpy(x3, x1); // 1
    fr_mul(x3, z);  // 11

    fr_cpy(x5, x3); // 11
    fr_mul(x5, z);  // 101

    fr_cpy(x7, x5); // 101
    fr_mul(x7, z);  // 111

    fr_cpy(z, x7);  // 111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 11100101

    fr_sqr(z);
    fr_mul(z, x5);  // 111001111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 111001111101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 111001111101101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 111001111101101101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 11100111110110110100111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 111001111101101101001110101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x3);  // 1110011111011011010011101010011

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 111001111101101101001110101001100101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x3);  // 1110011111011011010011101010011001010011

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 111001111101101101001110101001100101001100111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 1110011111011011010011101010011001010011001110101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 1110011111011011010011101010011001010011001110101111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 1110011111011011010011101010011001010011001110101111101

    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x1);  // 111001111101101101001110101001100101001100111010111110101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x1);  // 111001111101101101001110101001100101001100111010111110101001

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x3);  // 1110011111011011010011101010011001010011001110101111101010010000011

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x3);  // 11100111110110110100111010100110010100110011101011111010100100000110011

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 1110011111011011010011101010011001010011001110101111101010010000011001100111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 111001111101101101001110101001100101001100111010111110101001000001100110011100111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x3);  // 111001111101101101001110101001100101001100111010111110101001000001100110011100111011

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x1);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x1);  // 1110011111011011010011101010011001010011001110101111101010010000011001100111001110110000000100000001

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x3);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011

    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x1);  // 1110011111011011010011101010011001010011001110101111101010010000011001100111001110110000000100000001001101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x3);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 1110011111011011010011101010011001010011001110101111101010010000011001100111001110110000000100000001001101000011101100000000101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 1110011111011011010011101010011001010011001110101111101010010000011001100111001110110000000100000001001101000011101100000000101010100111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101001110111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101001110111101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101001110111101101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x1);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101001110111101101001

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 1110011111011011010011101010011001010011001110101111101010010000011001100111001110110000000100000001001101000011101100000000101010100111011110110100100000000101

    // replace x7 with x127

    fr_sqr(x7);
    fr_sqr(x7);
    fr_mul(x7, x3); // 11111
    fr_sqr(x7);
    fr_sqr(x7);
    fr_mul(x7, x3); // 1111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101001110111101101001000000001011111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 111001111101101101001110101001100101001100111010111110101001000001100110011100111011000000010000000100110100001110110000000010101010011101111011010010000000010111111111111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101001110111101101001000000001011111111111111100101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101001110111101101001000000001011111111111111100101101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 111001111101101101001110101001100101001100111010111110101001000001100110011100111011000000010000000100110100001110110000000010101010011101111011010010000000010111111111111111001011011111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x5);  // 111001111101101101001110101001100101001100111010111110101001000001100110011100111011000000010000000100110100001110110000000010101010011101111011010010000000010111111111111111001011011111111101

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 1110011111011011010011101010011001010011001110101111101010010000011001100111001110110000000100000001001101000011101100000000101010100111011110110100100000000101111111111111110010110111111111011111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101001110111101101001000000001011111111111111100101101111111110111111111111111

    // replace x127 with x255

    fr_sqr(x7);
    fr_mul(x7, x1); // 11111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 1110011111011011010011101010011001010011001110101111101010010000011001100111001110110000000100000001001101000011101100000000101010100111011110110100100000000101111111111111110010110111111111011111111111111111111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 111001111101101101001110101001100101001100111010111110101001000001100110011100111011000000010000000100110100001110110000000010101010011101111011010010000000010111111111111111001011011111111101111111111111111111111111111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 111001111101101101001110101001100101001100111010111110101001000001100110011100111011000000010000000100110100001110110000000010101010011101111011010010000000010111111111111111001011011111111101111111111111111111111111111111011111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 11100111110110110100111010100110010100110011101011111010100100000110011001110011101100000001000000010011010000111011000000001010101001110111101101001000000001011111111111111100101101111111110111111111111111111111111111111101111111111111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 1110011111011011010011101010011001010011001110101111101010010000011001100111001110110000000100000001001101000011101100000000101010100111011110110100100000000101111111111111110010110111111111011111111111111111111111111111110111111111111111111111111

    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_sqr(z);
    fr_mul(z, x7);  // 111001111101101101001110101001100101001100111010111110101001000001100110011100111011000000010000000100110100001110110000000010101010011101111011010010000000010111111111111111001011011111111101111111111111111111111111111111011111111111111111111111111111111
}

// vim: ts=4 et sw=4 si
