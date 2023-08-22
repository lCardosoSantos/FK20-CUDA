<!---
// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos
--->

Testing of FK20* functions
===============================================================================

This file document some characteristics of the tests of fk20* functions.

## FK20 Test
<!--- TODO: Write introduction ---> 

### Test functions
The test functions follow a common template, varying only the function and variables used:

```C
void input2output($type_t  $input[$x][$y], $type_t $output[$v][$w]) {
    cudaError_t err; //For checking sucessfull cuda execution
    bool pass = true;
    CLOCKINIT; //Macro to initialize code variables

    SET_SHAREDMEM($sharedmem, $functionName); //Not all functions need it.
    printf("=== RUN   %s\n", "functionName: $input -> $output");
    memset($res, 0xAA, $v * $w * sizeof($type_t)); 
    for(int testIDX=0; testIDX<=1; testIDX++){
        CLOCKSTART;
        $functionName<<<$rows, 256, $sharedmem>>>($res, ($type_t *)($input_l));
        CUDASYNC("$functionName");
        CLOCKEND;
        clearRes;
        $eq_wrapper<<<256, 32>>>(cmp, $x * $y, $res, ($type_t *)$output_l);
        CUDASYNC("$fr_eq_wrapper");
        if (testIDX == 0){ //check for normal test
            CMPCHECK($x * $y)
            PRINTPASS(pass);
            }
        else{ //check for false positive test
            NEGCMPCHECK($x*$y);
            NEGPRINTPASS(pass);
        }
        
        varMangle(($type_t*)$output_l, $v*$w, step);
    }
}
```
with the following metavariables:

| variable      | Description                                                        |
| ------------- | ------------------------------------------------------------------ |
| $input        | Input (or inputs) from the test vector                             |
| $input_l      | Local pointer to $input, with suffix `_l`                          |
| $output       | Output from the test vector                                        |
| $output_l     | Local pointer to $output, with suffix `_l`                         |
| $res          | Array of $type_t for temporary storage (`fr_tmp` or `g1p_tmp`)     |
| $rows         | Number of rows used in the test, determines gridSize.              |
| $sharedmem    | Optionally, value in bytes for dynamically allocated shared memory |
| $type_t       | variable type (`fr_t` or `g1p_t`)                                  |
| $v $w         | Dimensions of the output test vector                               |
| $x $y         | Dimensions of the input test vector                                |
| $functionName | tested function                                                    |
| $eq_wraper    | Comparison function for $type_t                                    |


The main parts of the test function are:

### False positive mitigation
```C
    memset($res, 0xAA, $v * $w * sizeof($type_t)); 
```

This initializes the array where the test function will write to with the bit pattern  `0xAA`. This works as a canary for faulty pointer arithmetic, for the case where the array happened to have a correct value, and the tested function fails and not overwrite it. This scenario can happen with a relevant probability, since some of the tested functions of FK20 produce results with long runs of identic bytes. This measures helps to avid false positives.

```C
    for(int testIDX=0; testIDX<=1; testIDX++){
        ...
        varMangle(($type_t*)$output_l, $v*$w, step);
    }
```
The tests are executed twice, with the second one expecting a failure. This is helped by the function `varMangle` which non destructively changes the input array. A second call to it resets the array back to it's correct state.

## Dynamic shared memory
```C
SET_SHAREDMEM($sharedmem, $functionName);
```

Some of the functions used in the FK20 computation use shared memory. While the ammount of shared memory allocated to those functions is constant, due to a limitation of CUDA it cannot be statically allocated. The same limitation also makes it necessary to set a proper CUDA function attribute. `SET_SHAREDMEM` is a macro that uses the proper `cudaFuncSetAttribute` call and check for any errors resulting fromm it. 
The sizes of shared memory are defined in `fk20.cuh`.

### Testing

The per-se testing happens in

```C
        CLOCKSTART;
        $functionName<<<$rows, 256, $sharedmem>>>($res, ($type_t *)($input_l));
        CUDASYNC("$functionName");
        CLOCKEND;
        clearRes;
        $eq_wrapper<<<256, 32>>>(cmp, $x * $y, $res, ($type_t *)$output_l);
        CUDASYNC("$eq_wrapper");
```

The macros `CLOCKSTART` and `CLOCKEND` implement a basic execution timer and reporting; `CUDASYNC` asserts that the computation on the GPU is finished; and `clearRes` zeros the comparison array.

The `$eq_wrapper` is a type-appropriate comparison function. The types used in FK20 to represent mathematical objects may have different bit values to represent the same object, hence the need to an arithmetic comparator. 

## FK20 Benchmark
### Benchmarking functions
The benchmark functions follow a common template, varying only the function and variables used:

```C++
void benchFunction(unsiged rows){
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds[NSAMPLES];
    float median;

    SET_SHAREDMEM($sharedmem, $functionName); //Not all functions need it.

    BENCH_BEFORE;
        functionName<<<rows, 256>>>($vars);
    BENCH_AFTER("polynomial -> tc");
}
```

There the macros `BENCH_BEFORE` and `BENCH_AFTER` are used to surround the function to be benchmarked. It will execute the function many times, record the runtime using `cudaEventRecord` and report the median, lowest, and highest execution times. Number of samples and rows are stored in globals from the command line args. 

### Auxiliary functions

Here are the auxiliary functions that set up the benchmark. These functions might be changed accordingly to new benchmarks added.

- `setupMemory`: Allocates dynamic unified memory for the inputs used in the benchmarked functions, as well as populate it with valid values.
- `freeMemory`: Free the memory allocated by `setupMemory`
- `preBenchTest`: This function serves two purposes: First, it runs a full FK20 computation, and checks it against a KAT. If this test fails, it will be reported, and normal testing will continue. The main objective of this module is to benchmark functions, not testing, but this serves as a canary and warning if something is not working as expected. And the second function is to allow the device to generate its bytecode and load it, with the effect of "spinning up the engines". In practical terms, it helps to remove initialization overheads.
- `printHeader`: Uses `cudaGetDeviceProperties` to report name and characteristics of the device, as a banner for the benchmarks.
<!---
## FK20_512 test
 TODO:

TODO: 
--->


