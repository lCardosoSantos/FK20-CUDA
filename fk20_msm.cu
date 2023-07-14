#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

////////////////////////////////////////////////////////////////////////////////

// fk20_msm(): toeplitz_coefficients_fft + xext_fft -> hext_fft

// parameters: 
// - in  xe_fft    array with dimensions [16][512]
// - in  tc_fft    array with dimensions [16][512]
// - out he_fft    array with dimensions [512]

__global__ void fk20_msm(g1p_t *he_fft, const fr_t *tc_fft, const g1p_t *xe_fft) {
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;  // k
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number

    g1p_t a0, a1, t;

    g1p_inf(a0);
    g1p_inf(a1);

    // move pointer for blocks
    he_fft += 512*bid;
    tc_fft += 16*512*bid;

    // MSM Loop
    for (int i=0; i<16; i++) {

        // Multiply and accumulate

        g1p_cpy(t, xe_fft[512*i+tid+0]);
        g1p_mul(t, tc_fft[512*i+tid+0]);
        g1p_add(a0, t);

        g1p_cpy(t, xe_fft[512*i+tid+256]);
        g1p_mul(t, tc_fft[512*i+tid+256]);
        g1p_add(a1, t);
    }

    // hext_fft = a0||a1
    // Store accumulators
    g1p_cpy(he_fft[tid+  0], a0);
    g1p_cpy(he_fft[tid+256], a1);
}

// vim: ts=4 et sw=4 si
