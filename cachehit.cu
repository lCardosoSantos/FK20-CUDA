#include <stdio.h>
#include <stdint.h>
#include "test.h"

__device__ void cacheHit(float z[], float x[], float y[], int c) {
    float z0 = z[0];    float x0 = x[0];    float y0 = y[0]; 
    float z1 = z[1];    float x1 = x[1];    float y1 = y[1]; 
    float z2 = z[2];    float x2 = x[2];    float y2 = y[2]; 
    float z3 = z[3];    float x3 = x[3];    float y3 = y[3]; 
    float z4 = z[4];    float x4 = x[4];    float y4 = y[4]; 
    float z5 = z[5];    float x5 = x[5];    float y5 = y[5]; 
    float z6 = z[6];    float x6 = x[6];    float y6 = y[6]; 
    float z7 = z[7];    float x7 = x[7];    float y7 = y[7]; 
    float z8 = z[8];    float x8 = x[8];    float y8 = y[8]; 
    float z9 = z[9];    float x9 = x[9];    float y9 = y[9]; 
    float z10 = z[10];  float x10 = x[10];  float y10 = y[10]; 
    float z11 = z[11];  float x11 = x[11];  float y11 = y[11]; 
    float z12 = z[12];  float x12 = x[12];  float y12 = y[12]; 
    float z13 = z[13];  float x13 = x[13];  float y13 = y[13]; 
    float z14 = z[14];  float x14 = x[14];  float y14 = y[14]; 
    float z15 = z[15];  float x15 = x[15];  float y15 = y[15]; 

    
        asm volatile(
            #include "asm.inc"
        :
        "=f"(z0), "=f"(z1), "=f"(z2), "=f"(z3), "=f"(z4), "=f"(z5), "=f"(z6), "=f"(z7), "=f"(z8), "=f"(z9), "=f"(z10), "=f"(z11), "=f"(z12), "=f"(z13), "=f"(z14), "=f"(z15)
        :
        "f"(x0), "f"(x1), "f"(x2), "f"(x3), "f"(x4), "f"(x5), "f"(x6), "f"(x7), "f"(x8), "f"(x9), "f"(x10), "f"(x11), "f"(x12), "f"(x13), "f"(x14), "f"(x15),
        "f"(y0), "f"(y1), "f"(y2), "f"(y3), "f"(y4), "f"(y5), "f"(y6), "f"(y7), "f"(y8), "f"(y9), "f"(y10), "f"(y11), "f"(y12), "f"(y13), "f"(y14), "f"(y15) 
        );


    z[0] = z0; 
    z[1] = z1; 
    z[2] = z2; 
    z[3] = z3; 
    z[4] = z4; 
    z[5] = z5; 
    z[6] = z6; 
    z[7] = z7; 
    z[8] = z8; 
    z[9] = z9; 
    z[10] = z10; 
    z[11] = z11; 
    z[12] = z12; 
    z[13] = z13; 
    z[14] = z14; 
    z[15] = z15; 

}

__global__ void cacheHitW(float z[], float x[], float y[], int c){
    for (int i=0; i<c; i++)
        cacheHit(z, x, y, c);
}


#define BENCH_BEFORE \
for(int i=0; i<NSAMPLES; i++){\
    cudaEventRecord(start)

#define COL(N) "\x1B["#N"G"

#define BENCH_AFTER(FNAME)\
    cudaEventRecord(stop); \
        err = cudaEventSynchronize(stop);\
        if (err != cudaSuccess) printf("%s:%d  Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));\
        cudaEventElapsedTime(&milliseconds[i], start, stop);\
    }\
    qsort(milliseconds, NSAMPLES, sizeof(milliseconds[0]), compare);\
    median = milliseconds[NSAMPLES/2];\
    printf(FNAME COL(25) " %8.6f ms [%8.6f - %8.6f]\n", median, milliseconds[0], milliseconds[NSAMPLES-1]);



__managed__ float z[16], x[16], y[16];
int compare(const void *  a, const void *  b);


int main(int argc, char const *argv[])
{
    #define NSAMPLES 1024
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds[NSAMPLES];
    float median;

    for (int i=0; i<16; i++){
        x[i]=i;
        y[i]=i*50.5;
    }

    BENCH_BEFORE;
    cacheHitW<<<1, 1>>>(z, x, y, 10000);
    BENCH_AFTER("cache");


    return 0;
}

int compare(const void *  a, const void *  b){
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}


/*
nvcc -rdc=true --std=c++14 --maxrregcount=128 --gpu-architecture=compute_80 --gpu-code=sm_86 cachehit.cu -o cachehit; ./cachehit
*/
