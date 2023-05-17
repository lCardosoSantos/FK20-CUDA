//Reads a input file and parses its data
#define  _GNU_SOURCE
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>
#include "parseFFTTest.h"

void freeffttest_t( ffttest_t* fftTest){
    // Because a part of the struct is dinamically allocated, you need to free a member of it
    // Ignore this function under penalty of memory leak
    free(fftTest->testCase);

}

void parsePoly(char *line, ffttest_t *fftTest, ssize_t line_read_size){
    //fprintf(stderr, "<%s>\n", __func__);
    char tmp[17];
    if(strncmp(line, "polynomial ", 11) != 0){
        printf("Fatal, malformed input: Expected polynomial keyword\n");
        exit(1);
    }

    ssize_t expectedLineSize = 11+(64*fftTest->polynomialLength)+fftTest->polynomialLength;
    if(line_read_size != expectedLineSize){
        printf("Fatal, malformed input: Expected %ld input size, read %ld\n", expectedLineSize, line_read_size);
        exit(1);
    }

    line += 11*sizeof(char);
    
    int testIdx=fftTest->nTest;
    fftTest->testCase = realloc(fftTest->testCase, (testIdx+1)*sizeof(struct FFTTestCase));
    memset(&(fftTest->testCase[testIdx]), 0x00, sizeof(struct FFTTestCase));
    fftTest->testCase[testIdx].idx = testIdx;

    for(int poly_i = 0; poly_i<fftTest->polynomialLength; poly_i++){
        for(int i=0; i<4; i++){
            strncpy(tmp, line,  16); tmp[16]='\0';
            fftTest->testCase[testIdx].polynomial[poly_i].word[3-i] = strtoul(tmp, NULL, 16);
            line+=16*sizeof(char);
        }

        line+=1*sizeof(char);
    }
}

void parsefft(char *line, ffttest_t *fftTest, unsigned char input){
    char tmp[17]; uint64_t val;
    int testIdx = fftTest->nTest;
    line = memchr(line, ' ', 32)+sizeof(char);
    for(int fft_i = 0; fft_i<fftTest->polynomialLength*2; fft_i++){
        for(int i=0; i<12; i++){
            strncpy(tmp, line,  16); tmp[16]='\0';
            val = strtoul(tmp, NULL, 16);
            if(input)
                fftTest->testCase[testIdx].fftInput[fft_i].word[i] = val;
            else
                fftTest->testCase[testIdx].fftOutput[fft_i].word[i] = val;
            line+=16*sizeof(char);
        }
        // Remove flag bits from coordinates
        if(input)
            fftTest->testCase[testIdx].fftInput[fft_i].word[11] &= (1ULL << 61) - 1;
        else
            fftTest->testCase[testIdx].fftOutput[fft_i].word[11] &= (1ULL << 61) - 1;

        line+=1*sizeof(char);
    }
}

void parsefftInput(char *line, ffttest_t *fftTest){
    //fprintf(stderr, "<%s>\n", __func__);
    if(strncmp(line, "fftTestInput", 12) != 0){
        printf("Fatal, malformed input: Expected fftTestInput keyword\n");
        exit(1);
    }

    parsefft(line, fftTest, 1);

}

void parsefftOutput(char *line, ffttest_t *fftTest){
    //fprintf(stderr, "<%s>\n", __func__);
    if(strncmp(line, "fftTestOutput", 13) != 0){
        printf("Fatal, malformed input: Expected fftTestInput keyword\n");
        exit(1);
    }

    parsefft(line, fftTest, 0);

}

void parseSetup(char *line, ffttest_t *fftTest){
    //fprintf(stderr, "<%s>\n", __func__);
    char tmp[32];
    if(strncmp(line, "setup ", 6) != 0){
        printf("Fatal, setup malformed >> %s", line);
        exit(1);
    }
    line += 6*sizeof(char);
    //printf("%s\n", line);
    
    for(int i=0; i<4; i++){
        strncpy(tmp, line,  16); tmp[16]='\0';
        fftTest->setup.word[3-i] = strtoul(tmp, NULL, 16);
        line+=16*sizeof(char);
    }

    //fprintf(stderr, "%lx %lx %lx %lx\n", fftTest->setup.word[0], fftTest->setup.word[1], fftTest->setup.word[2], fftTest->setup.word[3] );
}

void printTest(ffttest_t test){
    //gets out information on the read test, for debugging
    #define eprintf(fmt , ...) fprintf(stderr, fmt, ##__VA_ARGS__)

    eprintf("Tests read: %d \n", test.nTest);
    eprintf("Lenght of the polynomials: %d \n", test.polynomialLength);
    eprintf("Setup %016lx %016lx %016lx %016lx \n", test.setup.word[0], test.setup.word[1], test.setup.word[2], test.setup.word[3]);
    eprintf("\n");

    for(int i=0; i<test.nTest; i++){
        eprintf("test %02d \n", i);
        for(int ii=0; ii<4; ii++){
            eprintf(">tfftin[%d] ", ii);
            for (int j=0; j<6; j++) eprintf("%016lx ", test.testCase[i].fftInput[ii].word[j]); eprintf("\n");
            eprintf(">>fftot[%d] ", ii);
            for (int j=0; j<6; j++) eprintf("%016lx ", test.testCase[i].fftOutput[ii].word[j]); eprintf("\n");
        }
    }

    eprintf("\n");
}


ffttest_t parseFFTTest(const char *filename){
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == NULL){
        printf("Error while trying to open file %s\n", filename);
    }
    int ntests;
    char *line = NULL;
    size_t line_buff_size = 0;
    int line_count=0;
    ssize_t line_read_size = 0; //will infinite loop if this variable is not signed.

    //read first Line, expexted number of tests
    line_read_size = getline(&line, &line_buff_size, fp);
    ntests = atoi(line+7*sizeof(char));

    ffttest_t fftTest;
    fftTest.nTest=0; fftTest.polynomialLength=POLYLEN;
    fftTest.testCase = malloc(sizeof(struct FFTTestCase)*ntests);

    //read line, expected to be the setup
    line_read_size = getline(&line, &line_buff_size, fp);
    //printf("Read %ld chars in a %ld buff: %s", line_read_size, line_buff_size, line);
    parseSetup(line, &fftTest);

    do{
        line_read_size = getline(&line, &line_buff_size, fp);

        //stop reading at a hashmark
        if (line[0]=='#')
            break;
        if(line_read_size < 0)
            break;

        parsePoly(line, &fftTest, line_read_size);

        line_read_size = getline(&line, &line_buff_size, fp);
        if(line_read_size < 0){
            printf("fatal: No fftTestInput for Poly");
        }
            
        parsefftInput(line, &fftTest);

        line_read_size = getline(&line, &line_buff_size, fp);
        if(line_read_size < 0){
            printf("fatal: No fftTestOutput for Poly");
        }
        parsefftOutput(line, &fftTest);

        fftTest.nTest++;
    }while(1);

    fclose(fp);

    return fftTest;
}

#ifdef PARSEDEBUG
void main(){
    ffttest_t t = parseFFTTest("testFFT.in");
    printTest(t);
}
#endif
// compile for test
// gcc -o parseFFTTest.elf parseFFTTest.c -std=c99 -ggdb -g3 -DPARSEDEBUG; ./parseFFTTest.elf
