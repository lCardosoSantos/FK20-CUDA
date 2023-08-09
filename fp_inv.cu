// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include "fp.cuh"

/**
 * @brief  Calculates the multiplicative inverse of x and stores in z
 * 
 * * This function calculates the multiplicative inverse of the argument.
 * An integer a is the inverse of z if a*z mod r == 1
 * 
 * Normally, the inverse is found by using the Extended Euclidean Algorithm, to 
 * find integers (z,y) to satisfy the Bézout's identity:
 * a*z + r*y == gcd(a, r) == 1
 * which can be rewritten as:
 * az-1 == (-y)*m which follows that a*z mod r == 1. This approach has complexity 
 * in the order of O(log2(r)).
 * 
 * This implementation uses Euler's theorem, calculating the inverse as z^(phi(r)-1). 
 * where phi is Euler's totient function. For the special case where r is prime, 
 * phi(r) = r-1. Therefore, the inverse here is calculated as z^(r-2). 
 * Although this is asymptotically worse than EEA, this implementation avoid flow 
 * divergence and uses 279 squarings and 128 multiplications. 
 * Furthermore, since curve operations are done in projective coordinates, inversions
 * are needed only at the very end when projective coordinates are translated into 
 * affine coordinates.
 * 
 * @param[out] z 
 * @param[in] x 
 * @return void
 */
__device__ void fp_inv(fp_t &z, const fp_t &x) {

    // p-2 = 4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559785
    // = 5 * 5683 * 8572387 * 60565473370971138564533387 * 271298243114458311795492131869447332014235303245210008608904867493862992995991

    uint64_t x1[6], x3[6];

    // 5: 101

    fp_cpy(x1, x);
    fp_sqr(z, x);
    fp_sqr(z, z);
    fp_mul(x1, z, x1);

    // 5683: 1011000110011

    fp_sqr(z, x1);
    fp_mul(x3, z, x1);

    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(x1, z, x3);

    // 8572387: 100000101100110111100011
    //          11    1 11  11 11     11
    //                           11     

    fp_sqr(z, x1);
    fp_mul(x3, z, x1);

    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(x1, z, x3);

    // 60565473370971138564533387: 11001000011001001111001110001111101100000100111001111101010010111100111010110010001011
    //                             11  1    11  1  11    11    11  1 11     1  11   11  1 1 1  1 11    11  1 11  1   1 11
    //                                               11    1     11              1    11           11    1

    fp_sqr(z, x1);
    fp_mul(z, z, x1);
    fp_cpy(x3, z);

    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(x1, z, x3);

    // 271298243114458311795492131869447332014235303245210008608904867493862992995991

    fp_sqr(z, x1);
    fp_mul(x3, z, x1);

    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_sqr(z, z);
    fp_mul(z, z, x3);
    fp_sqr(z, z);
    fp_mul(z, z, x1);
}

// vim: ts=4 et sw=4 si
