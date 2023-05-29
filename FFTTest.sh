#!/bin/bash

N=16  # Default value

# Parse command line arguments
while getopts ":n:" opt; do
  case ${opt} in
    n)
      N=${OPTARG}
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Run CudaTest in parallel to generate test data
for ((i=1; i<=N; i++))
do
  file="fftdata.$i.in"
  if [ ! -f "$file" ]; then
    FK20Py/fk20_multi_CudaTest.py > $file &  # Run CudaTest in the background
  fi
done

# Wait for all parallel instances of CudaTest to finish
wait

echo "Running $i tests:"
# Run fffttest in series
for ((i=1; i<=N; i++))
do
  ./ffttest < "fftdata.$i.in"  # Run fffttest
done
