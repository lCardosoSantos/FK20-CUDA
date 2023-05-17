//Reads a input file and parses its data
#ifndef parseFFTTest_H
#define parseFFTTest_H

typedef struct uint768 {uint64_t word[12];} uint768_t;
typedef struct uint256 {uint64_t word[4];} uint256_t;

#define POLYLEN 512 //The test for GPU code is expected to have always the same input length

struct FFTTestCase{
    unsigned int idx;
    uint256_t polynomial[POLYLEN];
    uint768_t fftInput[POLYLEN*2];
    uint768_t fftOutput[POLYLEN*2];
};

typedef struct FFTTest{
    unsigned int nTest;
    unsigned int polynomialLength;
    uint256_t setup;
    struct FFTTestCase *testCase;
} ffttest_t;

extern void freeffttest_t( ffttest_t *fftTest);

extern ffttest_t parseFFTTest(const char *filename);

#endif