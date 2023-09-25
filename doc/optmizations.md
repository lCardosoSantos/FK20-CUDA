<!---
// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos
--->


The objective of this file is to track and document the high level optimizations used in the FK20 CUDA implementation

# FFT interleaved memory
Introduced in commit `831686f3143a38771ffda1eb1a9874d57aff33cc`

## Shared memory
CUDA memory hierarchy exposes two levels of cache to each  between it's registers and the Global memery. On the same level as the L1 cache it has a Shared memory, accessible to all threads in a Thread Block

This memory is used for inter-thread communication within a block, and is composed of 32 banks of memory, 4-byte wide. When writing to Shared Memory, each sucessive 4-byte word is written to a different bank. 
Shared access to this memory is issued in groups of 32 threads (warp).
- If **N** threads in the warp access different 4-byte words in the same bank, *the access are executed **serially** 
- If **N** threads access the same word in at the same time, it is results in a **multicast**

The following access contfigurations result in **no bank conflict**:

| Thread | Bank |
|--------|------|
| 0      | 0    |
| 1      | 1    |
| 2      | 2    |
| ...    | ...  |
| 31     | 31   |

Thread *n* doesn't need to access the bank *n%31*:

| Thread | Bank |
|--------|------|
| 0      | 15   |
| 1      | 16   |
| 2      | 6    |
| ...    | ...  |
| 31     | 6    |

This will generate a 2way conflict

| Thread | Bank |
|--------|------|
| 0      | 0    |
| 1      | 2    |
| 2      | 4    |
| 3      | 6    |
| 4      | 8    |
| ...    | ...  |
| 28     | 0    |
| 29     | 2    |
| 30     | 4    |
| 31     | 6    |

## Avoiding bank conflicts

The g1pFFT functions are executed on a array of `g1p_t[512]`, with size of 73728Bytes. A single `g1p` element has a size of `3*6*4` bytes (3 coordinates of type `fp_t`, which is composed by 6 QWORDS). Writing these elements in sharedmemory as is would result in hitting the whole width of the shared memory, by each thread. This in turn leads to a serialization of memory access.

Bank conflicts can then be solved by having each `g1p_t` in the array written as 4-Byte chunk in the shared memory, such that each thread in the warp will hit the same bank:

![[fftSharedMem.svg]]

Two functions are then used to write and read from shared memory into the local registers, by translating the index of the array element into the correct memory address. To simplify the process, the array and shared memory bank are cast into `uint32_t`.

With `smi` as the shared memory address, and `idx` as in index of the `g1p_t` element `g`, the position of each 32bit word of `g` is given by the following calculation:
1. `smi = (idx/32)*32*36` which moves the pointer to the correct row in the memory
2. `smi += idx%32` then moves the pointer for the correct column.
3. `smi += widx*32` moved the pointer down in the row, where `widx` in the index of the 32-bit word of `g` 

In `fk20_hext_fft2h_fft.cu`, two functions are defined to write to and read from the shared memory:
```C++
__device__ void wsm_g1p(unsigned index, const g1p_t *input);
__device__ void rsm_g1p(unsigned index, g1p_t *output);
```

In addition to this optimization, `hext_fft2h_fft` executes an IFT followed by a FFT.  Their implementation is based on `g1p_fft` and `g1p_ift`, with the ending of beginning changed as to avoid swapping elements of the array in memory. `ism_g1p(unsigned index)` can then be used to set `g1p_t` elements in the shared memory to the point-at-infinity.

## Root of unity

Beyond the workspace array, the FFT functions also access a precomputed table with the _root of unity_ values with type `fr_t[15]` and size of 16480 Bytes. 
A priori, the same optimization can be done to it, instead of having the valies in the global memory and cached. In addition to the interweaving, a "shiftrows" operation is applied to the table. If seen as a matrix of 32x17 elements, each row  $n$ is cyclically-shifted right by $n$ elements.
This is implemented in the following functions:
```C++
__device__ void wsm_fr(unsigned index, fr_t &input);
__device__ void rsm_fr(unsigned index, fr_t &output);
```

As off commit `38a0c4036d96dcef50eb5640c9bc7193df66b2e4` this feature is turned off by the macro `SHAREDMEMROOTS`. Increasing the amount of shared memory results in a smaller register file, that coupled with the binary-like access pattern and load costs results in no appreciable speedup, and introduces complexity to the code.

# Comb MSM
Introduced in commit `17e4600dbe5d878186f9c07ec0848b27d01cd819`

This optimization is divided in two main functions: `makelut` and `msm_comb`, and only applies to the case where 512 blocks/rows are executed.

## `fk20_msm_makelut`
This function is used to precompute the look up table used by `fk20_msm_comb`. Current implementation is single threaded, since this table is needed to be generated only once. The input to this function is the preprocessed commitment `xext_fft`, and outputs an array of $G_1$ points in **affine** representation, of type `g1a_t[16][512][256]`

## `fk20_msm_comb`
Perform the same operation as `fk20_msm`  ( `toeplitz_coefficients_fft + xext_fft -> hext_fft`).  
`fk20_msm` executes up to 512 blocks, which each block having 256 threads and processing a single polynomial. Conceptually, each block of threads forms a row in the computation matrix. There is no computation dependencies between the blocks.
`fk20_msm_comb` is also instantiated with 512blocks of 256 threads, but the computation is done in a collum-wise fashion. Since all the point multiplications in the same column uses the same element of `xext_fft`, this allows the use of a 8-way comb multiplication. 