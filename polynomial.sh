#! /bin/bash
echo \#include \"fr.cuh\"
echo
echo __managed__ fr_t
echo "polynomial[512*4096] = {"
for ((i=0; i<512; i++)); do
	echo // $i
	grep -A4096 polynomial test/fk20test-fib-1-$i.cu | tail -4096
done
echo "};"
