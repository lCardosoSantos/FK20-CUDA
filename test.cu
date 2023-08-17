// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#include <unistd.h>
#include "test.h"

__managed__ int stdout_isatty;

/**
 * @brief Sets a global variable to true if the STDOUT is a terminal.
 * Needs to be done like so because while a kernel is able to print to stdout,
 * it is unable to execute isatty.
 *
 * Currently used so piping to a file doesn't output terminal control characters.
 *
 */
void testinit() {
	stdout_isatty = isatty(fileno(stdout));
}

// vim: ts=4 et sw=4 si
