// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#include "fr.cuh"
#include "fp.cuh"
#include "g1.cuh"
#include "fk20.cuh"
#include "fk20test.cuh"

//debug macros for dumping elements to file

#define WRITEU64(writing_stream, var, nu64Elem) do{ \
    uint64_t *pointer = (uint64_t *)(*var); \
    for (int count=0; count<(nu64Elem); count++){ \
        fprintf(writing_stream,"%016lx\n",pointer[count]); \
    } \
}while(0)

#define WRITEU64TOFILE(fileName, var, nu64Elem) do{ \
    FILE * filepointer = fopen(fileName, "w");     \
    WRITEU64(filepointer, var, (nu64Elem));           \
    fclose(filepointer);                           \
}while(0) 

#define WRITEU64STDOUT(var, nu64Elem) do{ \
    uint64_t *pointer = (uint64_t *)(*var); \
    for (int count=0; count<(nu64Elem); count++){ \
        printf("%016lx\n",pointer[count]); \
    } \
}while(0)

// Testvector inputs

extern __managed__ g1p_t xext_fft[16][512];
extern __managed__ fr_t polynomial[512*4096];

// Intermediate values

extern __managed__ fr_t toeplitz_coefficients[512*16][512];
extern __managed__ fr_t toeplitz_coefficients_fft[512*16][512];
extern __managed__ g1p_t hext_fft[512*512];
extern __managed__ g1p_t h[512*512];

// Testvector output

extern __managed__ g1p_t h_fft[512*512];

// Workspace

static __managed__ uint8_t cmp[512*16*512];
static __managed__ fr_t fr_tmp_[512*16*512];
static __managed__ g1p_t g1p_tmp[512*512];
#define PTRN_G1PTMP memset(g1p_tmp, 0x88, 512*512*sizeof(g1p_t));
#define PTRN_FRTMP  memset(fr_tmp_, 0x88, 512*16*512*sizeof(fr_t));

//512 tests
void toeplitz_coefficients2toeplitz_coefficients_fft_512();
void h2h_fft_512();
void h_fft2h_512();
void hext_fft2h_512();
void hext_fft2h_fft_512();

void fk20_poly2toeplitz_coefficients_512(int execN);
void fk20_poly2hext_fft_512();
void fk20_poly2h_fft_512();
void fk20_msmloop_512();
void fk20_poly2toeplitz_coefficients_fft_test();


void fk20_poly2h_fft_512_(){
    //memory pattern
    memset(g1p_tmp, 0x88, 512*512*sizeof(g1p_t));
    
    //move pointers one foward
    fr_t  *p =  polynomial; //+ 4096*2;    
    g1p_t  *x =  (g1p_t *)xext_fft;   //actually constant for the 512 rows
    g1p_t  *h =  h_fft; //+ 512*2;//h_fft; //will fail at block zero if you move the pointers to block 1. Suggests KAT is wrong.

    //copy from test0 into test1;
    memcpy(polynomial+4096, polynomial, 4096*sizeof(fr_t));
    memcpy(h_fft+512, h_fft, 512*sizeof(g1p_t));
    
    cudaError_t err;
    bool pass = true;
    clock_t start, end;
    int nBlocks = 2; 

    err = cudaFuncSetAttribute(fk20_poly2h_fft, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN [%d]  %s\n", nBlocks, "fk20_poly2h_fft: polynomial -> h_fft");

    start = clock();
    fk20_poly2h_fft<<<nBlocks, 256, g1p_sharedmem>>>(g1p_tmp, p, (const g1p_t *)x);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2h_fft: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<nBlocks*512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<1, 32>>>(cmp, nBlocks*512, g1p_tmp, (g1p_t *)h);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<nBlocks*512; i++)
        if (cmp[i] != 1) {
            pass = false;
            printf(">>Breaks at idx %d \n", i); 
            break;
            
        }

    PRINTPASS(pass);
}


int main_(){
    fk20_poly2h_fft_512_();

    return 0;
}

int main() {
    
    //all tests
    toeplitz_coefficients2toeplitz_coefficients_fft_512();
    h2h_fft_512();
    h_fft2h_512();
    hext_fft2h_512();
    fk20_poly2toeplitz_coefficients_512(1);  
    fk20_poly2hext_fft_512(); //current work
    hext_fft2h_fft_512(); //TODO: Test Fails
    fk20_poly2h_fft_512(); //TODO: Test Fails
    fk20_msmloop_512(); //TODO: Test Fails
    fk20_poly2toeplitz_coefficients_fft_test(); //TODO: Test Fails
    //toeplitz_coefficients2toeplitz_coefficients_fft_512();
    //h2h_fft_512();
    //h_fft2h_512();
    //hext_fft2h_512();
    //fk20_poly2toeplitz_coefficients_512(512);
    //fk20_poly2hext_fft_512(); //TODO: Test Fails
    fk20_poly2h_fft_512();
    fk20_poly2h_fft_512();

    //hext_fft2h_fft_512(); //TODO: Test Fails
    //fk20_msmloop_512(); //TODO: Test Fails
    //fk20_poly2toeplitz_coefficients_fft_test(); //TODO: Test Fails

    
    
    /*
    //remove uncertainty
    for(int i=0; i<(512*16*512); i++){
        fr_tmp_[i][0]=1;
        fr_tmp_[i][1]=1;
        fr_tmp_[i][2]=1;
        fr_tmp_[i][3]=1;
    } 
    
    fk20_poly2h_fft_512();
    */
    return 0;
}

/*
Luan's notes
causes fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients to fail idx0201:
    toeplitz_coefficients -> toeplitz_coefficients_fft 

causes fk20_poly2h_fft: polynomial -> h_fft to fail (cudaErrorIllegalAddress)
    fr_fft: toeplitz_coefficients -> toeplitz_coefficients_fft

    g1p_fft: h -> h_fft

    g1p_ift: h_fft -> h

    g1p_ift: hext_fft -> h

    fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients

    fk20_poly2hext_fft: polynomial -> hext_fft


Some awk magix
awk 'getline p<f && p!=$0 {print "Line " NR ":" RS $0 RS p; exit}' f=file2 file1
*/

void toeplitz_coefficients2toeplitz_coefficients_fft_512(){
    PTRN_FRTMP;
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    printf("=== RUN   %s\n", "fr_fft: toeplitz_coefficients -> toeplitz_coefficients_fft");
    start = clock();
    fr_fft_wrapper<<<512*16, 256, fr_sharedmem>>>(fr_tmp_, (fr_t *)toeplitz_coefficients);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fr_fft_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512*16*512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "fr_eq_wrapper", cmp, 512, fr_tmp_, h_fft); fflush(stdout);

    fr_eq_wrapper<<<256, 32>>>(cmp, 512*16*512, fr_tmp_, (fr_t *)toeplitz_coefficients_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Check FFT result

    for (int i=0; pass && i<512*16*512; i++)
        if (cmp[i] != 1) {
            printf("FFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);
}

void h2h_fft_512(){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    err = cudaFuncSetAttribute(g1p_fft_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));


    printf("=== RUN   %s\n", "g1p_fft: h -> h_fft");
    start = clock();
    g1p_fft_wrapper<<<512, 256, g1p_sharedmem>>>(g1p_tmp, h);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess) printf("Error g1p_fft_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Clear comparison results

    for (int i=0; i<512*512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "g1p_eq_wrapper", cmp, 512, g1p_tmp, h_fft); fflush(stdout);

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512*512, g1p_tmp, h_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("Error g1p_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Check FFT result

    for (int i=0; pass && i<512*512; i++)
        if (cmp[i] != 1) {
            printf("FFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);

}

void h_fft2h_512(){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    err = cudaFuncSetAttribute(g1p_ift_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));


    printf("=== RUN   %s\n", "g1p_ift: h_fft -> h");

    start = clock();
    g1p_ift_wrapper<<<512, 256, g1p_sharedmem>>>(g1p_tmp, h_fft);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error g1p_ift_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512*512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "g1p_eq_wrapper", cmp, 512*512, g1p_tmp, h); fflush(stdout);

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512*512, g1p_tmp, h);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Check IFT result

    for (int i=0; pass && i<512*512; i++)
        if (cmp[i] != 1) {
            printf("IFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);

}

void hext_fft2h_512(){
    PTRN_G1PTMP;
    //TODO: Why the grid size is one in this function?
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    err = cudaFuncSetAttribute(g1p_ift_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "g1p_ift: hext_fft -> h");

    start = clock();
    g1p_ift_wrapper<<<1, 256, g1p_sharedmem>>>(g1p_tmp, hext_fft);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error g1p_ift_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512; i++)
        cmp[i] = 0;

    // printf("  %s(%p, %d, %p, %p)\n", "g1p_eq_wrapper", cmp, 512, g1p_tmp, h); fflush(stdout);

    g1p_eq_wrapper<<<8, 32>>>(cmp, 256, g1p_tmp, h);    // Note: h, not hext, hence 256, not 512

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    // Check IFT result

    for (int i=0; pass && i<256; i++)
        if (cmp[i] != 1) {
            printf("IFT error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);

}

void fk20_poly2toeplitz_coefficients_512(int execN){ 
    PTRN_FRTMP;
        char polyFilename [64];
        char fr_tmpFilename [64];
        char toeplitzFilename [64];
        //remove some uncertainty
        //memset(fr_tmp_, 1, 512*16*512*sizeof(fr_t)); //fr_tmp_[512*16*512];
        //for(int i=0; i<(512*16*512); i++){
        //    fr_tmp_[i][0]=0;//1;
        //    fr_tmp_[i][1]=0;//1;
        //    fr_tmp_[i][2]=0;//1;
        //    fr_tmp_[i][3]=0;//1;
        //} 


    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients: polynomial -> toeplitz_coefficients");
    start = clock();
        //sprintf(polyFilename,     "pol%d-%d.out", execN, 0  );
        //sprintf(fr_tmpFilename,   "tmp%d-%d.out", execN, 0  );
        //sprintf(toeplitzFilename, "toe%d-%d.out", execN, 0  );
        //WRITEU64TOFILE(polyFilename,     polynomial,            512*4096*4);
        //WRITEU64TOFILE(fr_tmpFilename,   fr_tmp_,                512*16*512*4);
        //WRITEU64TOFILE(toeplitzFilename, toeplitz_coefficients, 512*16*512*4);
    fk20_poly2toeplitz_coefficients<<<512, 256>>>(fr_tmp_, polynomial);
    err = cudaDeviceSynchronize();
        //sprintf(polyFilename,     "pol%d-%d.out", execN, 1  );
        //sprintf(fr_tmpFilename,   "tmp%d-%d.out", execN, 1  );
        //sprintf(toeplitzFilename, "toe%d-%d.out", execN, 1  );
        //WRITEU64TOFILE(polyFilename,     polynomial,            512*4096*4);
        //WRITEU64TOFILE(fr_tmpFilename,   fr_tmp_,                512*16*512*4);
        //WRITEU64TOFILE(toeplitzFilename, toeplitz_coefficients, 512*16*512*4);

    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2toeplitz_coefficients: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512*16*512; i++)
        cmp[i] = 0;

    fr_eq_wrapper<<<1, 32>>>(cmp, 512*16*512, fr_tmp_, (fr_t *)toeplitz_coefficients);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result
    
    for (int i=0; pass && i<512*16*512; i++)
        if (cmp[i] != 1) {
            printf("poly2toeplitz_coefficients error at idx 0x%04x\n", i);
            pass = false;
        }

    PRINTPASS(pass);
}

void fk20_poly2hext_fft_512(){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    pass = true;

    err = cudaFuncSetAttribute(fk20_poly2hext_fft, cudaFuncAttributeMaxDynamicSharedMemorySize, fr_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "fk20_poly2hext_fft: polynomial -> hext_fft");

    start = clock();
    fk20_poly2hext_fft<<<512, 256, g1p_sharedmem>>>(g1p_tmp, polynomial, (const g1p_t *)xext_fft);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2hext_fft: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512*512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<1, 32>>>(cmp, 512*512, g1p_tmp, (g1p_t *)hext_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<512*512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }

    PRINTPASS(pass);

}

void fk20_poly2h_fft_512(){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    printf("=== RUN   %s\n", "fk20_poly2h_fft: polynomial -> h_fft");

    start = clock();
    fk20_poly2h_fft(g1p_tmp, polynomial, (const g1p_t *)xext_fft, 512);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2h_fft: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512*512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<1, 32>>>(cmp, 512*512, g1p_tmp, (g1p_t *)h_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<512*512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }

    PRINTPASS(pass);
}

void hext_fft2h_fft_512(){
    PTRN_G1PTMP;
    cudaError_t err;
    bool pass = true;
    clock_t start, end;

    err = cudaFuncSetAttribute(fk20_hext_fft2h_fft, cudaFuncAttributeMaxDynamicSharedMemorySize, g1p_sharedmem);
    cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    printf("=== RUN   %s\n", "hext_fft2h_fft_512: hext_fft -> h_fft");

    start = clock();
    fk20_hext_fft2h_fft<<<512, 256, g1p_sharedmem>>>(g1p_tmp, hext_fft);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_hext_fft2h_fft: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512*512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<8, 32>>>(cmp, 512*512, g1p_tmp, h);   

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

    for (int i=0; pass && i<256; i++)
        if (cmp[i] != 1) {
            printf("fk20_hext_fft2h_fft error %d\n", i);
            pass = false;
        }

    PRINTPASS(pass);

}

void fk20_msmloop_512(){
    PTRN_G1PTMP;
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_msm: Toeplitz_coefficients+xext_fft -> hext_fft");
    memset(g1p_tmp,0xdeadbeef,512*512*sizeof(g1p_t)); //pattern on tmp dest
    start = clock();

    fk20_msm_xext_fftANDtoepliz_fft2hext_fft<<<512, 256>>>(g1p_tmp, (const fr_t*)toeplitz_coefficients_fft, (const g1p_t*)xext_fft);
    
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_msm: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512*512; i++)
        cmp[i] = 0;

    g1p_eq_wrapper<<<16, 32>>>(cmp, 512*512, g1p_tmp, (g1p_t *)hext_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error g1p_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<512*512; i++)
        if (cmp[i] != 1) {
            pass = false;
        }

    PRINTPASS(pass);

}

void fk20_poly2toeplitz_coefficients_fft_test(){
    PTRN_FRTMP;
    clock_t start, end;
    cudaError_t err;
    bool pass = true;

    printf("=== RUN   %s\n", "fk20_poly2toeplitz_coefficients_fft: polynomial -> toeplitz_coefficients_fft");
    memset(fr_tmp_, 0xdeadbeef,512*16*512*sizeof(fr_t)); //pattern on tmp dest.
    start = clock();
    fk20_poly2toeplitz_coefficients_fft<<<512, 256>>>(fr_tmp_, polynomial);
    err = cudaDeviceSynchronize();
    end = clock();

    if (err != cudaSuccess)
        printf("Error fk20_poly2toeplitz_coefficients_fft: %d (%s)\n", err, cudaGetErrorName(err));
    else
        printf(" (%.3f s)\n", (end - start) * (1.0 / CLOCKS_PER_SEC));

    // Clear comparison results

    for (int i=0; i<512*16*512; i++)
        cmp[i] = 0;

    fr_eq_wrapper<<<16, 256>>>(cmp, 512*16*512, fr_tmp_, (fr_t *)toeplitz_coefficients_fft);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) printf("Error fr_eq_wrapper: %d (%s)\n", err, cudaGetErrorName(err));

    // Check result

    for (int i=0; i<512*16*512; i++)
        if (cmp[i] != 1) {
            printf("poly2tc error %04x\n", i);
            pass = false;
            break;
        }

    PRINTPASS(pass);
}


// vim: ts=4 et sw=4 si
