#include <unistd.h>
#include "test.h"

__managed__ int stdout_isatty;

void testinit() {
	stdout_isatty = isatty(fileno(stdout));
}

// vim: ts=4 et sw=4 si
