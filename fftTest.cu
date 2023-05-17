//testing for the fk20, loselly based on the fk20test_kat.cu
#include <stdio.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "g1.cuh"
#include "fk20.cuh"

extern "C"{
    #include "parseFFTTest.h"
}

__managed__ g1p_t g1p_input[512], g1p_output[512], g1p_expected[512];

bool g1a_iszeroHost(const g1a_t &a) {
    return (a.x[5] | a.x[4] | a.x[3] | a.x[2] | a.x[1] | a.x[0] |
            a.y[5] | a.y[4] | a.y[3] | a.y[2] | a.y[1] | a.y[0]) == 0;
}

void g1p_fromG1aHost(g1p_t &p, const g1a_t &a) {
    if (g1a_iszeroHost(a)) {
        p = { { 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 } };
    }
    for(int i=0; i<6; i++) p.x[i]=a.x[i];
    for(int i=0; i<6; i++) p.y[i]=a.y[i];
    //fp_one(p.z);
    p.z[0]=1;
    for(int i=2; i<6; i++) p.z[i]=0;
}

void unpackffttest(ffttest_t testInputs, int testIDX, g1p_t g1p_input[512]){
    g1a_t tmp;
    //first, read the 256 fft input elements
    for(int argidx=0; argidx<256; argidx++){
        /* because of limitation in the API of BLST, the test-case generator only
         * has access to the affine representation of G1 elements -- where each ealement is represented as
         * two elements of fp. The g1p_fft uses the other representation, where an extra element is used. 
         * Notice that FFTTestCase.fftInputp is 
         */

        for(int j=0; j<6; j++){
        tmp.x[j] = testInputs.testCase[testIDX].fftInput[argidx].word[j];
        tmp.y[j] = testInputs.testCase[testIDX].fftInput[argidx].word[j+6];
        }
        //Convert these g1a to g1p
        g1p_fromG1aHost(g1p_input[argidx], tmp);
    }


    //the last 256 elements are zero at infinity due to the design of the reference python implementation
    g1p_t zinf = { { 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0 } };

    for(int i=256; i<512; i++)
        g1p_input[i] = zinf;

}

void FFTTest_random(){
    //generates tests from randomness
    return;
}

void FFTTest(){
    //uses tests picked from actual use cases, extracted from the instrumented python implementation
    const dim3 block(256,1,1);
    const dim3 grid(512,1,1);
    const size_t sharedmem = 73728; //72 KiB

    clock_t elapsedTime;

    //read data from testFFT.in using partseFFTTest
    const char inputFile[] = "testFFT.in";
    ffttest_t testInputs = parseFFTTest(inputFile); 
    if (testInputs.nTest == 0){
        exit(-1);
    } 
    else{
        fprintf(stderr, "<%s> Test inputs read: %d tests.\n", __func__, testInputs.nTest);
    }

    //convert testcase into g1p format
    unpackffttest(testInputs, 0, g1p_input);

    //Allocate memory
    const size_t fftsize = 512*sizeof(g1p_t);
    const size_t memsize = grid.x*fftsize;

    g1p_t *in, *out;

    cudaMallocManaged(&in,  memsize);
    cudaMallocManaged(&out, memsize);
    
    // Copy input to device
    for (int i=0; i<grid.x; i++) memcpy(in+i*512, g1p_input, fftsize);

    //run multi-fft
    elapsedTime = -clock();
    g1p_fft<<<grid, block, sharedmem>>> (out, in);
    cudaDeviceSynchronize();
    elapsedTime += clock();

    fprintf(stderr, "Kernel executed in %.5fs\n", elapsedTime * (1.0 / CLOCKS_PER_SEC) );
    //check for correctness, report errors
    fprintf(stderr, "Hello, I still don't do error checking, duuude\n");

    //dealocate
    cudaFree(in);
    cudaFree(out);
    freeffttest_t(&testInputs);

}


void init(){

}

int main(){
    init();
    printf("Debug\n");
    FFTTest();

    return 0;
}
/*
TODO:
    [ ] Make it so the parser returns stuff in the formats defined in g1.cuh
*/