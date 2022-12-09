# Test for cuda FFT

Proposal of interface for testing the CUDA implementation of FFT using the pythn reference code.
> To be defined: Default filenames.

## Input

### Generation method

The size of the polynomial is set as 256[^1]. A setup with be generated with a random secret, then `nTEST` random polinomials will be generated, and used with `fk.data_availabilty_using_fk20(polynomial, setup)` to generate the FFT test cases. For self-safety, the test case generation will also check it's own generate proofs. Alternativelly, the expected outputs of the fft can also be generated, allowing simple checks with `diff`.

[^1]: This size allows one to hardcode the root of unity and expanded root of unity. In this specific case, with len(poly)=256, len(expanded_root_of_unity) = 513

### File format
Input file to be parsed by the CUDA FFT code will be presented in the following text format

```
    0: setup <setup>
    i: polynomial <int> <int> ... <int> 
2*i+1: fftTestInput_i <P1> <P1> ... <P1>

```

`line0` contains the concrete values used to generate the code: The keyword `setup` followed by the 256 decimal integer used to generate the fk20 setup. Following, there will be two lines per case:
the fist line will have the keyword `polynomial` followed by 256 integers, representing the indexes of the poly, separated by spaces, and ending with a linefeed character. The following line will have the keyword `fftTestInput_i` where i is an integer with the test id. Following, separated by spaces, there will the the 512 curve points in affine format, following blst/zcash standard representation.

### P1 representation <a name="p1"></a>

The curve points are represented in affine format, consisting of a 96byte hexadecimal number, zero padded, without the `0x` prefix, and in lower case. 

From the ZCash BLS12-381 specification:

- Fq elements are encoded in big-endian form. They occupy 48 bytes in this form.
- Fq2 elements are encoded in big-endian form, meaning that the Fq2 element c0 + c1 * u is represented by the Fq element c1 followed by the Fq element c0. This means Fq2 elements occupy 96 bytes in this form.
- The group G1 uses Fq elements for coordinates. The group G2 uses Fq2 elements for coordinates.
- G1 and G2 elements can be encoded in uncompressed form (the x-coordinate followed by the y-coordinate) or in compressed form (just the x-coordinate). G1 elements occupy 96 bytes in uncompressed form, and 48 bytes in compressed form. G2 elements occupy 192 bytes in uncompressed form, and 96 bytes in compressed form.

The most-significant three bits of a G1 or G2 encoding should be masked away before the coordinate(s) are interpreted. These bits are used to unambiguously represent the underlying element:

- The most significant bit, when set, indicates that the point is in compressed form. Otherwise, the point is in uncompressed form.
- The second-most significant bit indicates that the point is at infinity. If this bit is set, the remaining bits of the group element's encoding should be set to zero.
- The third-most significant bit is set if (and only if) this point is in compressed form and it is not the point at infinity and its y-coordinate is the lexicographically largest of the two associated with the encoded x-coordinate.


### Argument for the format

The setup and polynomial are necessary information for regenerating the test case, hence it is important that they are kept. The use of keywords and space separation allows one to filter for only poly or inputs using `grep`. Inside a C program, it allows one to use the stdlib `strtok` to tokemize the input line. Lastly, using the blst/zcash representation allows easy use of those libraries for testing.

---

## Output

The CUDA implementation is expected to write to STDOUT[^2] the following format:

```
0: fftTestOutput_0 <P1> <P1> ... <P1>
1: fftTestOutput_1 <P1> <P1> ... <P1>
n: fftTestOutput_n <P1> <P1> ... <P1>
```

Where the line `n` is the result of the test case of same index. Point representation is the same as [input](#p1), space separated. Each line is preceeded by the keyword fftTestOutput_n where n is the test index. It is expected that each line will have 512 points.

[^2]: Outputing to STDOUT is better so you don't need to worry about handling an output file. For best, it would be good if debug messages are written to STDERR. But if not possible, the use of the keyword _only_ in the outputs makes grepping easy.