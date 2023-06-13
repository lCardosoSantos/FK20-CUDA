extern __managed__ fr_t polynomial[4096];
extern __managed__ g1p_t setup[4097];
extern __managed__ g1p_t xext_fft[16][512];
extern __managed__ fr_t toeplitz_coefficients[16][512];
extern __managed__ fr_t toeplitz_coefficients_fft[16][512];
extern __managed__ g1p_t hext_fft[512];
extern __managed__ g1p_t h[512];
extern __managed__ g1p_t h_fft[512];

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
    uint64_t *pointer = (uint64_t *)(*var); \
    for (int count=0; count<(nu64Elem); count++){ \
        printf("%016lx\n",pointer[count]); \
    } \
}while(0)

