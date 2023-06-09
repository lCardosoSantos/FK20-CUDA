#! /bin/bash

echo \#include \"fr.cuh\"
echo
echo __managed__ fr_t
echo "toeplitz_coefficients[512*16][512] = {"
for ((i=0; i<512; i++)); do
	echo // $i
	grep -A8224 toeplitz_coefficients\\\> test/fk20test-fib-1-$i.cu | tail -8224
done
echo "};"
