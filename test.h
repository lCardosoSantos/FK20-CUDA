#ifndef TEST_H
#define TEST_H

//pretty print
#define COLOR_RED     "\x1b[31m"
#define COLOR_GREEN   "\x1b[32m"
#define COLOR_RESET   "\x1b[0m"
#define COLOR_BOLD    "\x1b[1m"

#define PRINTPASS(pass) printf("--- %s\n", pass ? COLOR_GREEN "PASS" COLOR_RESET: COLOR_RED COLOR_BOLD "FAIL" COLOR_RESET);

#endif
