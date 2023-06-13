#! /bin/bash

echo \#include \"g1.cuh\"
echo
echo __managed__ g1p_t
echo "hext_fft[512*512] = {"

for ((i=0; i<512; i++)); do
	echo // $i
	grep -A2560 ^hext_fft test/fk20test-fib-1-$i.cu | tail -2560
done

echo "};"
