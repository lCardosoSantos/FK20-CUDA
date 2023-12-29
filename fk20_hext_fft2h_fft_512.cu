// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <cassert>
#include <cstdio>

#include "g1.cuh"
#include "fk20.cuh"
#include "g1p_fft_accel.cuh"
// #include "g1p_ptx.cuh"
// #define g1p_addsub(p, q) g1m(OP_ADDSUB, q, p, q, p);
// #include "fk20_hext_fft2h_fft_512.cuh"

// #define DEBUG

#ifdef DEBUG
    #define dprintf(...) fprintf(stderr, ##__VA_ARGS__)
#else
    #define dprintf(...)
#endif

#define cudaErrCheck(fmt, ...)                                                                                         \
    if (err != cudaSuccess)                                                                                            \
    printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)


void graphInit(g1p_t *h_fft);

__global__ void Stage(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r);

__global__ void fftStage0_1(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r);
__global__ void fftStageN(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r);
__global__ void fftStage1(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r);
__global__ void fftStage0(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r);

__global__ void iftStage0(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r);
__global__ void iftStage1(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r);
__global__ void iftStageN(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r, bool dbg);
__global__ void setRowInfinity(g1p_t h_fft[][512], unsigned COL);


/**
 * @brief Transpose a matrix of g1p of size 512x512 in place
 *        This funtion can be slow, as it will be used only while the MSM format
 *        is incompatible;
 * @param[in,out] M 
 */

void g1p512SquareTranspose(g1p_t *M){
    const size_t Msize = 512;
    g1p_t tmp;

    for(unsigned i=0; i<Msize; i++){
        for(unsigned j=i+1; j<Msize; j++){
            g1p_cpy(tmp, M[i*512+j]);
            g1p_cpy(M[i*512+j], M[j*512+i]);
            g1p_cpy(M[j*512+i], tmp);
        }
    }
}


/******************************************************************************/
bool GraphCreated = false;
g1p_t *graphPointer;
cudaGraph_t graph;
cudaGraphExec_t graphExec;


/**
 * @brief This function does not transpose inputs/outputs
 * 
 * @param h_fft 
 * @param hext_fft
 */
void fk20_hext_fft_2_h_fft_512(g1p_t *h_fft, const g1p_t *hext_fft){
    cudaStream_t zeroStream;
    cudaStreamCreate(&zeroStream);
    cudaError_t err; 

    if (h_fft != hext_fft){
    //copy from hext_fft to h_fft is addresses are diferent.
    memcpy(h_fft, hext_fft, 512*512*sizeof(g1p_t));
    dprintf("memory copied\n");
    }

    if(!GraphCreated || graphPointer != h_fft){
        graphInit(h_fft);
        GraphCreated = true;
        graphPointer = h_fft;
        dprintf("Graph init\n");
        #ifdef DEBUG
            cudaGraphDebugDotPrint(graph, "fftGraph.dot", cudaGraphDebugDotFlagsKernelNodeParams);
        #endif 
    }
    //Launch code

    dprintf("accel init\n");
    g1p_fft_accel_init();

    dprintf("Graph launch\n");
    err = cudaGraphLaunch(graphExec, zeroStream); 
    cudaErrCheck("graph launch");
    
    dprintf("Graph sync\n");
    err=cudaStreamSynchronize(zeroStream);
    cudaErrCheck("graph Sync");

    // cudaGraphExecDestroy(graphExec);
    // cudaGraphDestroy(graph);

}

/******************************************************************************/
void graphInit(g1p_t *h_fft){   //via api capture
    //consider that h_fft is populated with hext_fft and already transposed.
    const unsigned EventDepth = 18; //TODO: Update
    const unsigned nBlocks = 8; //empirical
    const unsigned nThreads = 512/nBlocks;
    const unsigned nCols = 256;
    unsigned w, l, r; 
    cudaError_t err;


    //init main graph Stream
    cudaStream_t sZero;
    cudaStreamCreate(&sZero);
    //init fork stream used in the 512 collums
    cudaStream_t sFFT[nCols];
    for(unsigned i=0; i<nCols; i++){
        cudaStreamCreate(&sFFT[i]);
    }

    //Events
    cudaEvent_t forkEvent, joinEvent[nCols], eventGraph[EventDepth][nCols*2];
    cudaEventCreate(&forkEvent);

    for (unsigned i=0; i<EventDepth; i++){
        for(unsigned j=0; j<nCols*2; j++){
            if(cudaEventCreate(&eventGraph[i][j]) != cudaSuccess)
                printf("failed to create event %u,%u", i, j); //TODO write cudaErrorCheck
        }
    }
    for (unsigned i=0; i<nCols; i++)
        cudaEventCreate(&joinEvent[i]);

    //Start graph capture
    cudaStreamBeginCapture(sZero, cudaStreamCaptureModeGlobal);

    /*------------------------------------------------------------------------*/
    //Fork graph
    cudaEventRecord(forkEvent, sZero);

    // //for graph debugging
    // struct eventGraphDebug{
    //     unsigned l, r;
    // } egd[EventDepth][nCols*2]={0};



#define eRECORD(DEPTH) cudaEventRecord(eventGraph[DEPTH][l], sFFT[COL]); \
                       cudaEventRecord(eventGraph[DEPTH][r], sFFT[COL]); 

#define eWAIT(DEPTH)  cudaStreamWaitEvent(sFFT[COL], eventGraph[DEPTH][l]); \
                      cudaStreamWaitEvent(sFFT[COL], eventGraph[DEPTH][r]); 
    // LAUNCH KERNELS
            //--------- IFT --------- 
    for(unsigned COL=0; COL<nCols; COL++){
        //conects each collum to the graph start;
        cudaStreamWaitEvent(sFFT[COL], forkEvent);
    }
    
    for(unsigned COL=0; COL<nCols; COL++){
        //Stage8
            w = (COL & 255) << 0;
            l = COL + (COL & -256U);
            r = l | 256;
            
            // iftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r, false);
            Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, 512-w, l, r);
            eRECORD(0);
    }

    for(unsigned COL=0; COL<nCols; COL++){
            //Stage7
            w = (COL & 127) << 1;
            l = COL + (COL & -128U);
            r = l | 128;
            eWAIT(0);
            // iftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r, false);
            Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, 512-w, l, r);
            eRECORD(1);
    }
    
    for(unsigned COL=0; COL<nCols; COL++){
            //Stage6
            w = (COL & 63) << 2;
            l = COL + (COL & -64U);
            r = l | 64;
            eWAIT(1);
            // iftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r, false);
            Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, 512-w, l, r);
            eRECORD(2);
    }
    for(unsigned COL=0; COL<nCols; COL++){
            //Stage5
            w = (COL & 31) << 3;
            l = COL + (COL & -32U);
            r = l | 32;
            eWAIT(2);
            // iftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r, false);
            Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, 512-w, l, r);
            eRECORD(3);
    }
    for(unsigned COL=0; COL<nCols; COL++){
            //Stage4
            w = (COL & 15) << 4;
            l = COL + (COL & -16U);
            r = l | 16;
            eWAIT(3);
            // iftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r, false);
            Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, 512-w, l, r);
            eRECORD(4);
    }
    for(unsigned COL=0; COL<nCols; COL++){
            //Stage3
            w = (COL & 7) << 5;
            l = COL + (COL & -8U);
            r = l | 8;
            eWAIT(4);
            // iftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r, false);
            Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, 512-w, l, r);
            eRECORD(5);
    }

    for(unsigned COL=0; COL<nCols; COL++){
            //Stage2
            w = (COL & 3) << 6;
            l = COL + (COL & -4U);
            r = l | 4;
            eWAIT(5);
            // iftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r, false);
            Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, 512-w, l, r);
            eRECORD(6);
    }

    for(unsigned COL=0; COL<nCols; COL++){
            //Stage1
            w = (COL & 1) << 0;
            l = COL + (COL & -2U);
            r = l | 2;
            
            eWAIT(6);
            iftStage1<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
            // Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, 512-w, l, r); //TODO: Error here
            eRECORD(7);
    }

    for(unsigned COL=0; COL<nCols; COL++){
            //Stage0
            w = 0;
            l = COL * 2;
            r = l | 1;

            eWAIT(7);
            // iftStage0<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
            Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, 512-w, l, r);
            eRECORD(8);
    }



    //------- ZeroHalf ------
    //similar to 
    // asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(secondHalfIndexes) : "r"(tid << (32-9)));
    // which is just every other index.
    for(unsigned COL = 0; COL<256; COL++){
        cudaStreamWaitEvent(sFFT[COL], eventGraph[8][COL * 2]);
        cudaEventRecord(eventGraph[9][COL * 2], sFFT[COL]); // mark the ones that don need a zero as done.
        setRowInfinity<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL * 2 + 1); 
        cudaEventRecord(eventGraph[9][COL * 2 + 1], sFFT[COL]);
    }

            //--------- FFT----------

    //// Stage 0
    for (unsigned COL = 0; COL < nCols; COL++) {
        w = 0;
        l = 2 * COL;
        r = l | 1;
        eWAIT(9);
        // fftStage0<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t(*)[512])h_fft, COL, w, l, r);
        Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);

        eRECORD(10);
    }
    //// Stage 1
    for (unsigned COL = 0; COL < nCols; COL++) {
        w = (COL & 1) << 7;
        l = COL + (COL & -2U);
        r = l | 2;
        eWAIT(10);
        // fftStage1<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t(*)[512])h_fft, COL, w, l, r);
        Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
        eRECORD(11);
    }
    for (unsigned COL = 0; COL < nCols; COL++) {
        //// Stage 2

        w = (COL & 3) << 6;
        l = COL + (COL & -4U);
        r = l | 4;
        eWAIT(11);
        // fftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t(*)[512])h_fft, COL, w, l, r);
        Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
        eRECORD(12);
    }
    for (unsigned COL = 0; COL < nCols; COL++) {
        //// Stage 3

        w = (COL & 7) << 5;
        l = COL + (COL & -8U);
        r = l | 8;
        eWAIT(12);
        // fftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t(*)[512])h_fft, COL, w, l, r);
        Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
        eRECORD(13);
    }
    //// Stage 4
    for (unsigned COL = 0; COL < nCols; COL++) {
        w = (COL & 15) << 4;
        l = COL + (COL & -16U);
        r = l | 16;
        eWAIT(13);
        // fftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t(*)[512])h_fft, COL, w, l, r);
        Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
        eRECORD(14);
    }
    //// Stage 5
    for (unsigned COL = 0; COL < nCols; COL++) {
        w = (COL & 31) << 3;
        l = COL + (COL & -32U);
        r = l | 32;
        eWAIT(14);
        // fftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t(*)[512])h_fft, COL, w, l, r);
        Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
        eRECORD(15);
    }
    //// Stage 6
    for (unsigned COL = 0; COL < nCols; COL++) {
        w = (COL & 63) << 2;
        l = COL + (COL & -64U);
        r = l | 64;
        eWAIT(15);
        // fftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t(*)[512])h_fft, COL, w, l, r);
        Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
        eRECORD(16);
    }
    //// Stage 7
    for (unsigned COL = 0; COL < nCols; COL++) {
        w = (COL & 127) << 1;
        l = COL + (COL & -128U);
        r = l | 128;
        eWAIT(16);
        // fftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t(*)[512])h_fft, COL, w, l, r);
        Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
        eRECORD(17);
    }
    //// Stage 8
    for (unsigned COL = 0; COL < nCols; COL++) {
        w = (COL & 255) << 0;
        l = COL + (COL & -256U);
        r = l | 256;
        eWAIT(17);
        // fftStageN<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t(*)[512])h_fft, COL, w, l, r);
        Stage<<<nBlocks, nThreads, 0, sFFT[COL]>>>((g1p_t (*)[512])h_fft, COL, w, l, r);
        // eRECORD(8);
    }

    //broadcast end of processing
    for(unsigned COL=0; COL < nCols; COL++){
        cudaEventRecord(joinEvent[COL], sFFT[COL]); //Join all streams to sZero
    }
    //Join graph
    for (unsigned i=0;i<nCols; i++)
        cudaStreamWaitEvent(sZero, joinEvent[i]);
    /*------------------------------------------------------------------------*/
    
    //End graph capture
    cudaStreamEndCapture(sZero, &graph);
    err = cudaGraphInstantiate(&graphExec, graph, 0);
    cudaErrCheck("graph instantiate");

    //destroy stream
    cudaStreamDestroy(sZero);
    for(unsigned i=0; i<nCols; i++){
        cudaStreamDestroy(sFFT[i]);
    }

    //destroy events
    cudaEventDestroy(forkEvent);
    for(unsigned i=0; i<nCols; i++) cudaEventDestroy(joinEvent[i]);

    for (unsigned i=0; i<EventDepth; i++){
        for(unsigned j=0; j<nThreads; j++){
            cudaEventDestroy(eventGraph[i][j]);
        }
    }

    // printf("Debug Event Graph l \n");
    // for(int j=0; j<16; j++){
    //     for(int i=0; i<9; i++){
    //         printf("%u; ", egd[i][j].l);
    //     }
    //     printf("\n");
    // }

    // printf("Debug Event Graph r \n");
    // for(int j=0; j<16; j++){
    //     for(int i=0; i<9; i++){
    //         printf("%u; ", egd[i][j].r);
    //     }
    //     printf("\n");
    // }

}


//for debug
#define TARGETCOL 1
/*                          FFT and IFT KERNELS                                */
__global__ void Stage(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r){
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);
    assert(blockDim.x * gridDim.x == 512);

    // unsigned tid = threadIdx.x; // Thread number
    // unsigned bid = blockIdx.x;  // Block number
    // unsigned idx = blockDim.x*bid+tid;
    g1p_fft_accel(h_fft[l], h_fft[r], w);

}


__global__ void fftStage0_1(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r){

    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);
    assert(blockDim.x * gridDim.x == 512);

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned idx = blockDim.x*bid+tid;
    __shared__ fr_t root;

    //stage 0
    g1p_addsub(h_fft[l][idx], h_fft[r][idx]); //TODO: g1m sub

    //stage 1
    w = (COL & 1) << 7;
    l = COL + (COL & -2U);
    r = l | 2;
    fr_cpy(root, fr_roots[w]);
    

    if (w) 
        g1p_mul(h_fft[r][idx], root);
    g1p_addsub(h_fft[l][idx], h_fft[r][idx]);

    // if(COL == TARGETCOL && idx==0){
    //     printf("%s returned", __func__);
    // }
}

__global__ void fftStageN(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r){
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);
    assert(blockDim.x * gridDim.x == 512);

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned idx = blockDim.x*bid+tid;
    __shared__ fr_t root;

    fr_cpy(root, fr_roots[w]);

    g1p_mul(h_fft[r][idx], root);
    g1p_addsub(h_fft[l][idx], h_fft[r][idx]);
    
}
__global__ void fftStage1(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r){
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);
    assert(blockDim.x * gridDim.x == 512);

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned idx = blockDim.x*bid+tid;
    __shared__ fr_t root;

    fr_cpy(root, fr_roots[w]);

    if (w) g1p_mul(h_fft[r][idx], root);
    g1p_addsub(h_fft[l][idx], h_fft[r][idx]); 
    
}
__global__ void fftStage0(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r){
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);
    assert(blockDim.x * gridDim.x == 512);

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned idx = blockDim.x*bid+tid;
    __shared__ fr_t root;

    fr_cpy(root, fr_roots[w]);

    // g1p_mul(h_fft[r][idx], root);
    g1p_addsub(h_fft[l][idx], h_fft[r][idx]);
    
}
__global__ void iftStage0(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r){
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);
    assert(blockDim.x * gridDim.x == 512);

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned idx = blockDim.x*bid+tid;
    
    // g1p_add(h_fft[l][idx], h_fft[r][idx]); //Not addsub, since the values in idxR will be discarted
    g1p_addsub(h_fft[l][idx], h_fft[r][idx]); 
    // if(idx==0 && COL == 0){
    //     g1p_print("lg ", h_fft[l][idx]);
    //     g1p_print("rg ", h_fft[r][idx]);
    // }

}

__global__ void iftStage1(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r){
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);
    assert(blockDim.x * gridDim.x == 512);

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned idx = blockDim.x*bid+tid;
    __shared__ fr_t rootl, rootr;
    
    fr_cpy(rootl, fr_roots[513]);
    fr_cpy(rootr, fr_roots[513+w]);
    
    //stage 1
    g1p_addsub(h_fft[l][idx], h_fft[r][idx]);
    g1p_mul(h_fft[l][idx], rootl);
    g1p_mul(h_fft[r][idx], rootr);
}
__global__ void iftStageN(g1p_t h_fft[][512], unsigned COL,  unsigned w,  unsigned l,  unsigned r, bool dbg=0){
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);
    assert(blockDim.x * gridDim.x == 512);

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned idx = blockDim.x*bid+tid;
    
    __shared__ fr_t root;

    fr_cpy(root, fr_roots[512-w]);

    g1p_addsub(h_fft[l][idx], h_fft[r][idx]);
    g1p_mul(h_fft[r][idx], root);
    
        // if(COL==0 && blockIdx.x == 0 && threadIdx.x ==0 & dbg==1){
        //     printf("COL %u, w %u, l %u, r %u\n", COL, w, l, r);
        //     g1p_print("l", h_fft[l][idx]);
        //     g1p_print("r", h_fft[r][idx]);
        // }

}

/**
 * @brief Set elements all the g1p elements in h_fft[col] to inf.
 * 
 * @param h_fft array g1p_t[512][512]
 * @param COL 
 * @return __global__ 
 */
__global__ void setRowInfinity(g1p_t h_fft[][512], unsigned COL){
    assert(gridDim.y  ==   1);
    assert(gridDim.z  ==   1);
    assert(blockDim.y ==   1);
    assert(blockDim.z ==   1);
    assert(blockDim.x * gridDim.x == 512);

    unsigned tid = threadIdx.x; // Thread number
    unsigned bid = blockIdx.x;  // Block number
    unsigned idx = blockDim.x*bid+tid;
    g1p_inf(h_fft[COL][idx]);
}
