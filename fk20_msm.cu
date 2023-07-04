#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

////////////////////////////////////////////////////////////////////////////////
// fk20_msm_xext_fftANDtoepliz_fft2hext_fft(): toeplitz_coefficients_fft_l + xext_fft -> hext_fft_l
// TODO: Update
// parameters: 
// - in  toeplitz_coefficients_fft_l    array with dimensions [512]
// - in  xext_fft                       array with dimensions [16][512]
// - out hext_fft_l                     array with dimensions [16][512]
// Note: shared memory is used both in MSM loop and FFTs, without conflict
////TODO: add to 512 tests

__global__ void fk20_msm_xext_fftANDtoepliz_fft2hext_fft(g1p_t *hext_fft_l, const fr_t *toeplitz_coefficients_fft_l, const g1p_t *xext_fft) {
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

    //move pointer for blocks
    hext_fft_l += 512*bid;
    fr_t *frToep_fft = (fr_t *)(toeplitz_coefficients_fft_l+16*512*bid); //g1p_mul does not allow for const input.
    xext_fft += 16*512*bid;


    // MSM Loop
    for (int i=0; i<16; i++) {

        //fr points to toeplitz_fft.
        // Multiply and accumulate

        g1p_cpy(t, xext_fft[512*i+tid+0]);
        g1p_mul(t, frToep_fft[512*i+tid]);
        __syncthreads();
        g1p_add(a0, t);

        g1p_cpy(t, xext_fft[512*i+tid+256]);
        g1p_mul(t, frToep_fft[512*i+tid+256]);
        __syncthreads();
        g1p_add(a1, t);
    }
    //hext_fft_l = a0||a1
    // Store accumulators
    g1p_cpy(hext_fft_l[tid+  0], a0);
    g1p_cpy(hext_fft_l[tid+256], a1);

}