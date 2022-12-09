#include <cuda.h>
#include <stdio.h>
#include <inttypes.h>

int main(int argc, char **argv)
{
    cudaError_t cudaError;

    if (argc != 3)
        return -1;

    FILE *infile = fopen(argv[1], "r");

    if (infile == NULL)
        return -2;

    FILE *outfile = fopen(argv[2], "w");

    if (outfile == NULL)
    {
        fclose(infile);
        return -3;
    }

    // Read input header

    unsigned rows = 0;

    fscanf(infile, "%u", &rows);

    printf("%d row%s\n", rows, rows==1 ? "": "s");

    // Allocate memory

    uint64_t *input = NULL;

    cudaError = cudaMallocManaged(&input, rows*256*16*(4+18));

    fprintf(stderr, "%p, %d\n", input, cudaError);

    for (unsigned i=0; i<rows*256*16*(4+18); i++)
    {
        input[i] = i;
        if (input[i] != i)
        {
            fprintf(stderr, "input[] failed test %u\n", i);
            return -1;
        }
    }

    // Read AoS input data (each array element contains a scalar and a point)

    for (unsigned i=0; i<rows; i++) // 1 per SM
    {
        for (unsigned j=0; j<256; j++)  // 8 warps * 32 threads
        {
            for (unsigned k=0; k<16; k++)   // 1 thread
            {
                uint64_t s[4], p[18];

                fscanf(
                    infile,
                    " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64
                    " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64
                    " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64
                    " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64 " %" SCNx64,
                    &s[3], &s[2], &s[1], &s[0],
                    &p[17], &p[16], &p[15], &p[14], &p[13], &p[12],
                    &p[11], &p[10], &p[ 9], &p[ 8], &p[ 7], &p[ 6],
                    &p[ 5], &p[ 4], &p[ 3], &p[ 2], &p[ 1], &p[ 0]
                );

                printf(
                    "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 " * ("
                    "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 ", "
                    "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 ", "
                    "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 ")\n",
                    s[3], s[2], s[1], s[0],
                    p[17], p[16], p[15], p[14], p[13], p[12],
                    p[11], p[10], p[ 9], p[ 8], p[ 7], p[ 6],
                    p[ 5], p[ 4], p[ 3], p[ 2], p[ 1], p[ 0]
                );

                // Store scalar and point in input array
                for (int l=0; l< 4; l++) input[l + 0 + 22*(k + 16*(j + 256*i))] = s[l];
                for (int l=0; l<18; l++) input[l + 4 + 22*(k + 16*(j + 256*i))] = p[l];
            }
        }
    }

#ifndef NDEBUG
    for (unsigned i=0; i<rows; i++) // 1 per SM
    {
        for (unsigned j=0; j<256; j++)  // 8 warps * 32 threads
        {
            for (unsigned k=0; k<16; k++)   // 1 thread
            {
                uint64_t s[4], p[18];

                // Read scalar and point from input array
                for (int l=0; l< 4; l++) s[l] = input[l + 0 + 22*(k + 16*(j + 256*i))];
                for (int l=0; l<18; l++) p[l] = input[l + 4 + 22*(k + 16*(j + 256*i))];

                printf(
                    "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 " * ("
                    "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 ", "
                    "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 ", "
                    "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 ")\n",
                    s[3], s[2], s[1], s[0],
                    p[17], p[16], p[15], p[14], p[13], p[12],
                    p[11], p[10], p[ 9], p[ 8], p[ 7], p[ 6],
                    p[ 5], p[ 4], p[ 3], p[ 2], p[ 1], p[ 0]
                );
            }
        putchar('\n');
        }
    }
#endif

    // Process inputs

//  fk20(rows, input, output);

    // Write output file

    fclose(infile);
    fclose(outfile);

    return 0;
}

// vim: ts=4 et sw=4 si
