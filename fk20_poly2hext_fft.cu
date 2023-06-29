#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

static __device__ fr_t fr_tmp[512*512];     // 16 KiB memory per threadblock

////////////////////////////////////////////////////////////////////////////////

// fk20_poly2hext_fft(): polynomial + xext_fft -> hext_fft

// parameters:
// - in  xext_fft   array with 16*512 elements, computed using fk20_setup2xext_fft()
// - in  polynomial array with 16*512*gridDim.x elements
// - out hext_fft   array with    512*gridDim.x elements

// Note: shared memory is used both in MSM loop and FFTs, without conflict

__global__ void fk20_poly2hext_fft(g1p_t *hext_fft, const fr_t *polynomial, const g1p_t xext_fft[8192]) {

    // gridDim.x is the number of rows
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;  // k
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    // Accumulators and temporaries in registers or local
    // (thread-interleaved global) memory

    g1p_t a0, a1, t;

    g1p_inf(a0);
    g1p_inf(a1);

    polynomial += 4096 * bid;
    hext_fft += 512 * bid;

    fr_t *fr = fr_tmp + 512 * bid;

    // MSM Loop

    for (int i=0; i<16; i++) {

        // Copy from the polynomial into half of the coefficient array

        unsigned src = 16*tid + 15 - i;
        unsigned dst = (tid+257) % 512;

        if (tid > 0)
            fr_cpy(fr[dst], polynomial[src]);
        else
            fr_zero(fr[dst]);

        __syncthreads();

        // Zero the other half of coefficients before FFT

        fr_zero(fr[tid+1]);

        // Compute FFT

        __syncthreads();
        fr_fft(fr, fr);
        __syncthreads();

        // Multiply and accumulate

        g1p_cpy(t, xext_fft[512*i + tid + 0]);
        g1p_mul(t, fr[tid]);
        __syncthreads();
        g1p_add(a0, t);

        g1p_cpy(t, xext_fft[512*i + tid + 256]);
        g1p_mul(t, fr[tid+256]);
        __syncthreads();
        g1p_add(a1, t);
    }

    // Store accumulators

    g1p_cpy(hext_fft[tid+  0], a0);
    g1p_cpy(hext_fft[tid+256], a1);
}

// vim: ts=4 et sw=4 si
