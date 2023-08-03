
#include <bits/getopt_core.h>
#include <cstring>
#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include <unistd.h>

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

extern __managed__ fr_t polynomial[4096];
extern __managed__ g1p_t setup[4097];
extern __managed__ g1p_t xext_fft[16][512];
extern __managed__ fr_t toeplitz_coefficients[16][512];
extern __managed__ fr_t toeplitz_coefficients_fft[16][512];
extern __managed__ g1p_t hext_fft[512];
extern __managed__ g1p_t h[512];
extern __managed__ g1p_t h_fft[512];


#include "test.h"

bool pass; 
static int NSAMPLES = 5;

#ifdef DEBUG
    #define DPRINTF(fmt, ...) fprintf(stderr, "[debug] %s:%d " fmt "\n", __FILE__, __LINE__,  ##__VA_ARGS__)
#else
    #define DPRINTF(fmt, ...)
#endif
/**************************** Workspace variables *****************************/
fr_t  *b_polynomial = NULL; //min[4096]; max[512*4096]
g1p_t *b_xext_fft = NULL; //min[16][512]; max[16][512];
fr_t  *b_toeplitz_coefficients = NULL; //min[16][512]; max [512*16][512];
fr_t  *b_toeplitz_coefficients_fft = NULL; //min[16][512]; max [512*16][512];
g1p_t *b_hext_fft = NULL; //min[512]; max [512*512];
g1p_t *b_h = NULL; //min[512]; max [512*512];
g1p_t *b_h_fft = NULL; //min[512]; max [512*512];

const size_t g1p_sharedmem = 512 * 3 * 6 * 8; // 512 points * 3 residues/point * 6 words/residue * 8 bytes/word = 72 KiB
const size_t fr_sharedmem = 512 * 4 * 8;      // 512 residues * 4 words/residue * 8 bytes/word = 16 KiB

//workspace
fr_t  *b_fr_tmp;
g1p_t *b_g1p_tmp;


/**************************** Preprocessing macros ****************************/
// The necessary shared memory is larger than what we can statically allocate, hence it is allocated dynamically in the
// kernel call. Because cuda, we need to set the maximum allowed size using this macro.
#define SET_SHAREDMEM(SZ, FN)                                                                                          \
    err = cudaFuncSetAttribute(FN, cudaFuncAttributeMaxDynamicSharedMemorySize, SZ);                                   \
    cudaDeviceSynchronize();                                                                                           \
    if (err != cudaSuccess)                                                                                            \
        printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

//TODO: Check if memcpy was sucessfull
#define COPYMANY(DEST, SRC, SIZE, NCOPIES, TYPE)                                                     \
        for(int counter=0; counter<NCOPIES; counter++) memcpy(DEST+counter*SIZE, SRC, SIZE*sizeof(TYPE));

// Syncronizes the Device, making sure that the kernel has finished the execution. Checks for any errors, and report if
// errors are found.
#define CUDASYNC(fmt, ...)                                                                                             \
    err = cudaDeviceSynchronize();                                                                                     \
    if (err != cudaSuccess)                                                                                            \
    printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)

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
    printf(FNAME COL(25) " %8.3f ms [%8.3f - %8.3f]\n", median, milliseconds[0], milliseconds[NSAMPLES-1]);


/********************************* Prototypes *********************************/
void setupMemory(unsigned rows);
void freeMemory();

bool preBenchTest(int rows);
void benchFull(int rows);
void benchSteps(unsigned rows);
void benchModules(unsigned rows);

int compare(const void *  a, const void *  b);
void printHeader(unsigned rows);

int main(int argc, char **argv) {
    unsigned rows = 32;
    NSAMPLES = 7;
    int opt;

    while((opt = getopt(argc, argv, "r:s:h")) != -1){
        switch (opt) {
            case 'r':
                rows = abs(atoi(optarg));
                rows = rows>512?512:rows; 
                break;
            case 's':
                NSAMPLES = abs(atoi(optarg));
                break;
            case 'h':
                printf("Usage: %s [-r rows] [-s NSAMPLES] [-h]\n", argv[0]);
                printf("Options:\n");
                printf("  -r #     Set the number of rows (default: %d)\n", rows);
                printf("  -s #     Set the number of samples (default: %d)\n", NSAMPLES);
                printf("  -h       Display this help information\n");
                return 0;
            case '?':
                if (optopt == 'r' || optopt == 's')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else 
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            default:
                return 1;
        }
    }

    printHeader(rows);

    setupMemory(rows);
    bool pass = preBenchTest(rows);
    if(!pass){
        printf("WARNING: An error was detected during the pre-benchmark test! Continuing... \n");
    }

    benchFull(rows);
    benchSteps(rows);
    benchModules(rows);
    freeMemory();
    return 0;
}

__managed__ uint8_t cmp[16*512];
bool preBenchTest(int rows){
    cudaError_t err;
    bool pass = true;
    // Setup
    //SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    DPRINTF("Pre-bench test %d rows ", rows); fflush(stdout);
    
        fk20_poly2toeplitz_coefficients<<<rows, 256>>>(b_fr_tmp, b_polynomial);
        fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(b_fr_tmp, b_fr_tmp);  
        fk20_msm<<<rows, 256>>>(b_g1p_tmp, b_fr_tmp,  (g1p_t *)xext_fft);
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(b_g1p_tmp, b_g1p_tmp);
        fk20_hext2h<<<rows, 256>>>(b_g1p_tmp);
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(b_g1p_tmp, b_g1p_tmp);

    clearRes;
    g1p_eq_wrapper<<<16, 32>>>(cmp, rows*512, b_g1p_tmp, b_h_fft);
    CUDASYNC("g1p_eq_wrapper");
    CMPCHECK(rows*512);
    #ifdef DEBUG
    PRINTPASS(pass);
    #endif
    return pass;
}

void benchFull(int rows){
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds[NSAMPLES];
    float median;

    // Setup

    //SET_SHAREDMEM(fr_sharedmem,  fr_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    printf("\n=== Test without stalling on Device\n");fflush(stdout);

    BENCH_BEFORE;
        fk20_poly2toeplitz_coefficients<<<rows, 256>>>(b_fr_tmp, b_polynomial);
        fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(b_fr_tmp, b_fr_tmp);  
        fk20_msm<<<rows, 256>>>(b_g1p_tmp, b_fr_tmp,  (g1p_t *)xext_fft);
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(b_g1p_tmp, b_g1p_tmp);
        fk20_hext2h<<<rows, 256>>>(b_g1p_tmp);
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(b_g1p_tmp, b_g1p_tmp);
    BENCH_AFTER("FK20");

}

void benchSteps(unsigned rows){
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds[NSAMPLES];
    float median;

    printf("\n=== Testing FK20 individual steps\n");

    SET_SHAREDMEM(g1p_sharedmem, g1p_fft_wrapper);
    SET_SHAREDMEM(g1p_sharedmem, g1p_ift_wrapper);

    
    BENCH_BEFORE;
        fk20_poly2toeplitz_coefficients<<<rows, 256>>>(b_fr_tmp, b_polynomial);
    BENCH_AFTER("polynomial -> tc");

    BENCH_BEFORE;
        fr_fft_wrapper<<<rows*16, 256, fr_sharedmem>>>(b_fr_tmp, b_fr_tmp);  
    BENCH_AFTER("tc -> tc_fft");

    BENCH_BEFORE;
        fk20_msm<<<rows, 256>>>(b_g1p_tmp, b_fr_tmp,  (g1p_t *)xext_fft);
    BENCH_AFTER("tc_fft -> hext_fft (msm)");

    BENCH_BEFORE;
        g1p_ift_wrapper<<<rows, 256, g1p_sharedmem>>>(b_g1p_tmp, b_g1p_tmp);
    BENCH_AFTER("hext_fft -> hext");

    BENCH_BEFORE;
        fk20_hext2h<<<rows, 256>>>(b_g1p_tmp);
    BENCH_AFTER("hext -> h");

    BENCH_BEFORE;
        g1p_fft_wrapper<<<rows, 256, g1p_sharedmem>>>(b_g1p_tmp, b_g1p_tmp);
    BENCH_AFTER("h -> h_fft");
}

void benchModules(unsigned rows){
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds[NSAMPLES];
    float median;

    printf("\n=== Testing FK20 components\n"); //The components you see in fk20test.cu

    SET_SHAREDMEM(g1p_sharedmem, fk20_hext_fft2h_fft)

    //Not used right now, may be usefull for future optimizations
    // BENCH_BEFORE;
    // fk20_hext_fft2h_fft<<<rows, 256, g1p_sharedmem>>>(b_g1p_tmp, b_hext_fft);
    // BENCH_AFTER("fk20_hext_fft2h_fft");

    BENCH_BEFORE;
    fk20_poly2hext_fft<<<rows, 256, fr_sharedmem>>>(b_g1p_tmp, b_polynomial, (const g1p_t *)b_xext_fft);
    BENCH_AFTER("fk20_poly2hext_fft");

    BENCH_BEFORE;
    fk20_poly2h_fft(b_g1p_tmp, b_polynomial, (const g1p_t *)xext_fft, rows);
    BENCH_AFTER("fk20_poly2h_fft");


}

void setupMemory(unsigned rows){
    //allocate memory and copy relevant data from the test vector
    //check, error on more than 193rows
    cudaError_t err;
    #define MALLOCSYNC(fmt, ...) \
        if (err != cudaSuccess)                                                                                            \
        printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)

    err = cudaMallocManaged(&b_polynomial, rows*4096*sizeof(fr_t)); 
          MALLOCSYNC("b_polynomial");
    err = cudaMallocManaged(&b_xext_fft, 16*512*sizeof(g1p_t)); //size not dependant on number of rows
          MALLOCSYNC("id");
    //err = cudaMallocManaged(&b_toeplitz_coefficients, rows*16*512*sizeof(fr_t));
    //      MALLOCSYNC("id");
    //err = cudaMallocManaged(&b_toeplitz_coefficients_fft, rows*16*512*sizeof(fr_t));
    //      MALLOCSYNC("id");
    err = cudaMallocManaged(&b_hext_fft, rows*512*sizeof(g1p_t));
          MALLOCSYNC("b_hext_fft");
    //err = cudaMallocManaged(&b_h, rows*512*sizeof(g1p_t));
    //      MALLOCSYNC("id");
    err = cudaMallocManaged(&b_h_fft, rows*512*sizeof(g1p_t));
          MALLOCSYNC("b_h_fft");
    err = cudaMallocManaged(&b_g1p_tmp, rows*512*sizeof(g1p_t));
          MALLOCSYNC("b_g1p_tmp");
    err = cudaMallocManaged(&b_fr_tmp, rows*16*512*sizeof(fr_t));
          MALLOCSYNC("b_fr_tmp");


    //copy data
    COPYMANY(b_polynomial, polynomial, 4096, rows, fr_t);
    COPYMANY(b_xext_fft, xext_fft, 16*512, 1, g1p_t);
    //COPYMANY(b_toeplitz_coefficients, toeplitz_coefficients, 16*512, rows, fr_t);
    //COPYMANY(b_toeplitz_coefficients_fft, toeplitz_coefficients_fft, 16*512, rows, fr_t);
    COPYMANY(b_hext_fft, hext_fft, 512, rows, g1p_t);
    //COPYMANY(b_h, h, 512, rows, g1p_t);
    COPYMANY(b_h_fft, h_fft, 512, rows, g1p_t);


    DPRINTF("Memory setup done");
}

void freeMemory(){
    //frees the pointers allocated by setupMemory
    cudaFree(b_polynomial);
    cudaFree(b_xext_fft);
    cudaFree(b_toeplitz_coefficients);
    cudaFree(b_toeplitz_coefficients_fft);
    cudaFree(b_hext_fft);
    cudaFree(b_h);
    cudaFree(b_h_fft);
    DPRINTF("Allocated memory freed");
}


void printHeader(unsigned rows){
    int kb=1<<10, mb=1<<20;

    printf("===  FK20 Benchmark: %d thread blocks\n", rows);
    printf("     Reporting median of %d executions as median [lowest | highest] \n", NSAMPLES);

    int devCount;
    cudaGetDeviceCount(&devCount);

    for(int i=0; i<devCount; i++){
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        printf("     GPU %d: %s: compute capability %d.%d\n", i, props.name, props.major, props.minor);
        printf("     Global memory:   %luMB\n", props.totalGlobalMem / mb);
        printf("     Shared memory:   %luKB\n", props.sharedMemPerBlock / kb);
        printf("     Constant memory: %luKB\n", props.totalConstMem / kb);
        printf("     Registers per block : %d\n", props.regsPerBlock);
        printf("     Multiprocessor count : %d\n\n", props.multiProcessorCount);

        printf("     Warp size:         %d\n", props.warpSize);
        printf("     Threads per block: %d\n", props.maxThreadsPerBlock);
        printf("     Max block dimensions: [ %d, %d, %d ]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
        printf("     Max grid dimensions:  [ %d, %d, %d ]\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
        printf("\n");
    }
}


int compare(const void *  a, const void *  b){
  //for usage with the qsort() function of stdlib
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}