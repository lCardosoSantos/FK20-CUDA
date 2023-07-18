#ifndef TEST_H
#define TEST_H
// Macros used in the testing functions

// pretty print
#include <memory>
#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_RESET "\x1b[0m"
#define COLOR_BOLD "\x1b[1m"

#define PRINTPASS(pass)                                                                                                \
    printf("--- %s\n", pass ? COLOR_GREEN "PASS" COLOR_RESET : COLOR_RED COLOR_BOLD "FAIL" COLOR_RESET);

#define NEGPRINTPASS(pass)                                                                                                \
    printf("--- %s (intentional error detected)\n", pass ? COLOR_RED COLOR_BOLD "FAIL" COLOR_RESET : COLOR_GREEN "PASS" COLOR_RESET);


// debug macros for dumping elements to file
#define WRITEU64(writing_stream, var, nu64Elem)                                                                        \
    do {                                                                                                               \
        uint64_t *pointer = (uint64_t *)(*var);                                                                        \
        for (int count = 0; count < (nu64Elem); count++) {                                                             \
            fprintf(writing_stream, "%016lx\n", pointer[count]);                                                       \
        }                                                                                                              \
    } while (0)

#define WRITEU64TOFILE(fileName, var, nu64Elem)                                                                        \
    do {                                                                                                               \
        FILE *filepointer = fopen(fileName, "w");                                                                      \
        WRITEU64(filepointer, var, (nu64Elem));                                                                        \
        fclose(filepointer);                                                                                           \
    } while (0)

// sadly cuda doesn't allow fprintf inside a kernel, so printfItis.
#define WRITEU64STDOUT(var, nu64Elem)                                                                                  \
    do {                                                                                                               \
        uint64_t *pointer = (uint64_t *)(var);                                                                         \
        for (int count = 0; count < (nu64Elem); count++) {                                                             \
            printf("%016lx ", pointer[count]);                                                                         \
            if (count % 6 == 5)                                                                                        \
                printf("\n");                                                                                          \
        }                                                                                                              \
    } while (0)

// Syncronizes the Device, making sure that the kernel has finished the execution. Checks for any errors, and report if
// errors are found.
#define CUDASYNC(fmt, ...)                                                                                             \
    err = cudaDeviceSynchronize();                                                                                     \
    if (err != cudaSuccess)                                                                                            \
    printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__)

// The necessary shared memory is larger than what we can statically allocate, hence it is allocated dynamically in the
// kernel call. Because cuda, we need to set the maximum allowed size using this macro.
#define SET_SHAREDMEM(SZ, FN)                                                                                          \
    err = cudaFuncSetAttribute(FN, cudaFuncAttributeMaxDynamicSharedMemorySize, SZ);                                   \
    cudaDeviceSynchronize();                                                                                           \
    if (err != cudaSuccess)                                                                                            \
        printf("Error cudaFuncSetAttribute: %s:%d, error %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));

// Clears the array used on the comparison kernel.
#define clearRes                                                                                                       \
    for (int i = 0; i < 16 * 512; i++)                                                                                 \
        cmp[i] = 0;                                                                                                    \
    pass = true;

#define clearRes512                                                                                                    \
    for (int i = 0; i < rows * 16 * 512; i++)                                                                          \
        cmp[i] = 0;                                                                                                    \
    pass = true;

#define CLOCKINIT clock_t start, end
#define CLOCKSTART start = clock()
#define CLOCKEND                                                                                                       \
    end = clock();                                                                                                     \
    printf(" (%.1f ms)\n", (end - start) * (1000. / CLOCKS_PER_SEC))

// Check if the array generated by the fr_eq() and g1p_eq() kernels have a failure, then report.
#define CMPCHECK(LENGTH)                                                                                               \
    for (int i = 0; pass && i < LENGTH; i++) {                                                                         \
        if (cmp[i] != 1) {                                                                                             \
            printf("%s:%d %s() error idx %d...\n", __FILE__, __LINE__, __func__, i);                                   \
            pass = false;                                                                                              \
            break;                                                                                                     \
        }                                                                                                              \
    }

// Check if the array generated by the fr_eq() and g1p_eq() kernels have a failure
#define NEGCMPCHECK(LENGTH)                                                                                               \
    for (int i = 0; pass && i < LENGTH; i++) {                                                                         \
        if (cmp[i] != 1) {                                                                                             \
            pass = false;                                                                                              \
            break;                                                                                                     \
        }                                                                                                              \
    }

#endif
