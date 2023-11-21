#ifndef FK20_HEXT_FFT2H_FFT_512_CUH
#define FK20_HEXT_FFT2H_FFT_512_CUH

void g1p512SquareTranspose(g1p M[512][512]);
void fk20_hext_fft_2_h_fft_512(g1p_t *h_fft, const g1p_t *hext_fft);


#endif
