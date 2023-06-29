#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

static __device__ fr_t fr_tmp[512*512];    // 16 KiB memory per threadblock
static __device__ g1p_t g1p_tmp[512*512];  // 72 KiB memory per threadblock

////////////////////////////////////////////////////////////////////////////////

// fk20_poly2toeplitz_coefficients(): polynomial -> toeplitz_coefficients

// parameters:
// - in  polynomial array with 4096*gridDim.x elements
// - out toeplitz_coefficients array with 8192*gridDim.x elements

__global__ void fk20_poly2toeplitz_coefficients(fr_t *toeplitz_coefficients, const fr_t *polynomial) {

    // gridDim.x is the number of rows
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    polynomial += 4096 * bid;
    toeplitz_coefficients += 8192 * bid;

    for (int i=0; i<16; i++) {

        // Copy from the polynomial into half of the coefficient array

        unsigned src = tid*16+15-i;
        unsigned dst = (tid+257)%512 + 512*i;

        if (tid > 0)
            fr_cpy(toeplitz_coefficients[dst], polynomial[src]);
        else
            fr_zero(toeplitz_coefficients[dst]);

        __syncthreads();

        // Zero the other half of coefficients before FFT

        fr_zero(toeplitz_coefficients[512*i+tid+1]);
    }
}

// vim: ts=4 et sw=4 si
