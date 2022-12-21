//Reads a input file and parses its data
#define  _GNU_SOURCE
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>

#define POLYLEN 4//512 //The test for GPU code is expected to have always the same input lenght

typedef struct uint768 {uint64_t word[12];} uint768_t;
typedef struct uint256 {uint64_t word[4];} uint256_t;

struct FFTTestCase{
    unsigned int idx;
    uint256_t polynomial[POLYLEN];
    uint768_t fftInputp[POLYLEN*2];
};

typedef struct FFTTest{
    unsigned int nTest;
    unsigned int polynomialLenght;
    uint256_t setup;
    struct FFTTestCase *testCase;
} ffttest_t;


void freeffttest_t( ffttest_t* fftTest){

}

void parsePoly(char *line, ffttest_t *fftTest, ssize_t line_read_size){
    char tmp[16];
    if(strncmp(line, "polynomial ", 11) != 0){
        printf("Fatal, malformed input: Expected polynomial keyword\n");
        exit(1);
    }

    ssize_t expectedLineSize = 11+(64*fftTest->polynomialLenght)+fftTest->polynomialLenght;
    if(line_read_size != expectedLineSize){
        printf("Fatal, malformed input: Expected %ld input size, read %ld\n", expectedLineSize, line_read_size);
        exit(1);
    }

    line += 11*sizeof(char);
    
    int testIdx=fftTest->nTest;
    fftTest->testCase = realloc(fftTest->testCase, (testIdx+1)*sizeof(struct FFTTestCase));
    memset(&(fftTest->testCase[testIdx]), 0x00, sizeof(struct FFTTestCase));
    fftTest->testCase[testIdx].idx = testIdx;

    for(int poly_i = 0; poly_i<fftTest->polynomialLenght; poly_i++){
        for(int i=0; i<4; i++){
            strncpy(tmp, line,  16);
            fftTest->testCase[testIdx].polynomial[poly_i].word[3-i] = strtoul(tmp, NULL, 16);
            line+=16*sizeof(char);
        }

        line+=1*sizeof(char);
    }
}

void parsefftInput(char *line, ffttest_t *fftTest){
    char tmp[16];

    if(strncmp(line, "fftTestInput", 12) != 0){
        printf("Fatal, malformed input: Expected fftTestInput keyword\n");
        exit(1);
    }

    int testIdx = fftTest->nTest;
    line = memchr(line, ' ', 32)+sizeof(char);
    for(int fft_i = 0; fft_i<fftTest->polynomialLenght*2; fft_i++){
        for(int i=0; i<12; i++){
            strncpy(tmp, line,  16);
            fftTest->testCase[testIdx].fftInputp[fft_i].word[11-i] = strtoul(tmp, NULL, 16);
            line+=16*sizeof(char);
        }

        line+=1*sizeof(char);
    }

}

void parseSetup(char *line, ffttest_t *fftTest){
    char tmp[32];
    if(strncmp(line, "setup ", 6) != 0){
        printf("Fatal, setup malformed >> %s", line);
        exit(1);
    }
    line += 6*sizeof(char);
    //printf("%s\n", line);
    
    for(int i=0; i<4; i++){
        strncpy(tmp, line,  16);
        fftTest->setup.word[3-i] = strtoul(tmp, NULL, 16);
        line+=16*sizeof(char);
    }

    //printf("%lx %lx %lx %lx", fftTest->setup.word[0], fftTest->setup.word[1], fftTest->setup.word[2], fftTest->setup.word[3] );
}

ffttest_t parseFFTTest(char *filename){
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == NULL){
        printf("Error while trying to open file %s\n", filename);
    }

    ffttest_t fftTest;
    fftTest.nTest=0; fftTest.polynomialLenght=POLYLEN;
    fftTest.testCase = malloc(sizeof(struct FFTTestCase));

    char *line = NULL;
    size_t line_buff_size = 0;
    int line_count=0;
    ssize_t line_read_size = 0; //will infinite loop if this variable is not signed.

    //read first line, expected to be the setup
    line_read_size = getline(&line, &line_buff_size, fp);
    //printf("Read %ld chars in a %ld buff: %s", line_read_size, line_buff_size, line);
    parseSetup(line, &fftTest);

    do{
        line_read_size = getline(&line, &line_buff_size, fp);
        if(line_read_size < 0)
            break;
        parsePoly(line, &fftTest, line_read_size);

        line_read_size = getline(&line, &line_buff_size, fp);
        if(line_read_size < 0){
            printf("fatal: No fftTestInput for Poly");
        }
            
        parsefftInput(line, &fftTest);
        fftTest.nTest++;

        //stop reading at a hashmark
        if (line[0]=='#')
            break;
        if(line_read_size < 0)
            break;
        }while(1);

    fclose(fp);
}


void main(){
    parseFFTTest("testFFT.in");
}

// compile for test
// gcc -o parseFFTTest.elf parseFFTTest.c -std=c99 -ggdb -g3; ./parseFFTTest.elf