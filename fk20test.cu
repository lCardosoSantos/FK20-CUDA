// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"
#include "fk20_testvector.cuh"

int main() {

    testinit();

    FK20TestFFT();
    FK20TestPoly();
    //FK20TestFFTRand(NULL); 
        // Luan's note: This function here hasn't been updated in a while. 
        // The scope of it is now covered by the 512 row tests, but can be useful
        // for debugging future optimizations.

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//Useful functions for the falsifiability tests
////////////////////////////////////////////////////////////////////////////////


/**
 * @brief swap elements at positions multiple of step. Nondestructive, call
 * a second time to undo the changes
 * 
 * @param[out] target Pointer to array
 * @param[in] size length of the array
 * @param[in] step distance between elements swapped.
 */
void varMangle(fr_t *target, size_t size, unsigned step){
    fr_t tmp;
    if (target == NULL || size <= 0 || step <= 0)
        return;

    for (int i = 0; i < size; i += step) {
        if (i + step < size){
            memcpy(tmp, target+i, sizeof(fr_t));
            memcpy(target+i, target+i+1, sizeof(fr_t));   
            memcpy(target+i+1, tmp, sizeof(fr_t));   
        }
    }
    
}

/**
 * @brief swap elements at positions multiple of step. Nondestructive, call
 * a second time to undo the changes
 * 
 * @param[out] target Pointer to array
 * @param[in] size length of the array
 * @param[in] step distance between elements swapped.
 */
void varMangle(g1p_t *target, size_t size, unsigned step){
    g1p_t tmp;
    if (target == NULL || size <= 0 || step <= 0)
        return;

    for (int i = 0; i < size; i += step) {
        if (i + step < size){
            memcpy(&tmp, target+i, sizeof(g1p_t));
            memcpy(target+i, target+i+1, sizeof(g1p_t));   
            memcpy(target+i+1, &tmp, sizeof(g1p_t));   
        }
    }
    
}

// vim: ts=4 et sw=4 si
