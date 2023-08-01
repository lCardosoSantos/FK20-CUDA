// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"

// Testvector inputs

extern __managed__ fr_t polynomial[512*4096];

// Testvector output

extern __managed__ fr_t toeplitz_coefficients_fft[512*16][512];

// Workspace

static __managed__ uint8_t cmp[512*16*512];
static __managed__ fr_t fr_tmp[512*16*512];

int main(int argc, char **argv) {

    testinit();

    int rows = 2;

    if (argc > 1)
        rows = atoi(argv[1]);

    if (rows < 1)
        rows = 1;

    if (rows > 512)
        rows = 512;

    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    //////////////////////////////////////////////////

    for (int i=0; i<5; i++) {

        //////////////////////////////////////////////////

        pass = true;

        printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients_fft: polynomial -> toeplitz_coefficients_fft");

        start = clock();
        fk20_poly2toeplitz_coefficients_fft<<<512, 256>>>(fr_tmp, polynomial);
        err = cudaDeviceSynchronize();
        end = clock();

        if (err != cudaSuccess)
            printf("Error fk20_poly2toeplitz_coefficients_fft: %d (%s)\n", err, cudaGetErrorName(err));
        else
            printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

        // Clear comparison results

        for (int i=0; i<512*16*512; i++)
            cmp[i] = 0;

        fr_eq_wrapper<<<16, 256>>>(cmp, 512*16*512, fr_tmp, (fr_t *)toeplitz_coefficients_fft);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) printf("Error fr_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

        // Check result

        for (int i=0; pass && i<512*16*512; i++)
            if (cmp[i] != 1) {
                printf("poly2tc error %04x\n", i);
                pass = false;
            }

        PRINTPASS(pass);

        //////////////////////////////////////////////////
    }
    return 0;
}

// vim: ts=4 et sw=4 si
