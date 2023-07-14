#ifndef TEST_H
#define TEST_H
//Macros used in the testing functions  

//pretty print
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_RESET   "\x1b[0m"
#define COLOR_BOLD    "\x1b[1m"

#define PRINTPASS(pass) printf("--- %s\n", pass ? COLOR_GREEN "PASS" COLOR_RESET: COLOR_RED COLOR_BOLD "FAIL" COLOR_RESET);

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

//sadly cuda doesn't allow fprintf inside a kernel, so printfItis.
#define WRITEU64STDOUT(var, nu64Elem) do{ \
    uint64_t *pointer = (uint64_t *)(var); \
    for (int count=0; count<(nu64Elem); count++){ \
        printf("%016lx ",pointer[count]); \
        if (count%6==5) printf("\n"); \
    } \
}while(0)

#define CUDASYNC(fmt, ...) err = cudaDeviceSynchronize(); \
                           if (err != cudaSuccess) \
                                printf("%s@%d: " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)

#define SET_SHAREDMEM(SZ, FN) \
    err = cudaFuncSetAttribute(FN, cudaFuncAttributeMaxDynamicSharedMemorySize, SZ); \
    cudaDeviceSynchronize(); \
    if (err != cudaSuccess) printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

#define clearRes   for (int i=0; i<16*512; i++) cmp[i] = 0; \
                   pass=true;
#define rows 1

#define CLOCKINIT clock_t start, end
#define CLOCKSTART start=clock()
#define CLOCKEND end = clock();\
                 printf(" (%.1f ms) \n", (end - start) * (1000. / CLOCKS_PER_SEC))




#endif
