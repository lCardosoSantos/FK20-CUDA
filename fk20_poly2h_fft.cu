#include <time.h>

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

#include "fk20test.cuh"
#include "fk20_testvector.cuh"

#define ROWS 512

#define SET_SHAREDMEM(SZ, FN) \
    err = cudaFuncSetAttribute(FN, cudaFuncAttributeMaxDynamicSharedMemorySize, SZ); \
    cudaDeviceSynchronize(); \
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

#define CUDASYNC     err = cudaDeviceSynchronize(); \
                     if (err != cudaSuccess) printf("Error: %d (%s)\n", err, cudaGetErrorName(err))

static __managed__ fr_t fr[ROWS*16*512]; // 256 KiB per threadblock
static __managed__ g1p_t g1p[ROWS*512];  // 72 KiB per threadblock

////////////////////////////////////////////////////////////////////////////////

__global__ void fk20_hext2h(g1p_t *h) {
    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    h += 512*bid;
    g1p_inf(h[256+tid]);
}

////////////////////////////////////////////////////////////////////////////////
// fk20_poly2h_fft(): polynomial + xext_fft -> h_fft
// This is the full execution of FK20. 
// parameters:
// - in  xext_fft   array with 16*512 elements, computed using fk20_setup2xext_fft()
// - in  polynomial array with 16*512*rows elements
// - in  rows       number of rows to process in one kernel launch
// - out h_fft      array with    512*rows elements

__host__ void fk20_poly2h_fft(g1p_t *h_fft, const fr_t *polynomial, const g1p_t xext_fft[8192], unsigned rows) {
    cudaError_t err;

    // Setup

    SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    //SET_SHAREDMEM(fr_sharedmem,  fk20_msm);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    // polynomial -> tc
    fk20_poly2toeplitz_coefficients<<<rows, 256, fr_sharedmem>>>(fr, polynomial);
    CUDASYNC;

    // tc -> tc_fft
    for(int i=0; i<16; i++){
        fr_fft_wrapper<<<rows, 256, fr_sharedmem>>>(fr+512*i, fr+512*i);
    }
    CUDASYNC;

    // tc_fft -> hext_fft
    fk20_msm<<<rows, 256>>>(g1p, fr, xext_fft);
    CUDASYNC;

    // hext_fft -> hext
    g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(g1p, g1p);
    CUDASYNC;

    // hext -> h
    fk20_hext2h<<<rows, 256>>>(g1p);
    CUDASYNC;
    
    // h -> h_fft
    g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(h_fft, g1p);
    CUDASYNC;
}
// vim: ts=4 et sw=4 si
