// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"
#include "fk20_testvector.cuh"

int main() {
    FK20TestPoly();
    FK20TestFFT();
    FK20TestFFTRand(NULL);

    return 0;
}

//int main() {
//    FK20TestPoly();
//    return 0;
//}
// vim: ts=4 et sw=4 si

/*

//zero fr_tmp
printf(">>>>\n");
for(int i=0; i<(512*16); i++){
    fr_tmp[i][0]=0;
    fr_tmp[i][1]=0;
    fr_tmp[i][2]=0;
    fr_tmp[i][3]=0;
} 

*/