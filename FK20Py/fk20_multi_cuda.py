#!/usr/bin/python3

from py_ecc import optimized_bls12_381 as b
from fft import fft
import kzg_proofs
import sys
from kzg_proofs import (
    MODULUS,
    check_proof_multi,
    generate_setup,
    commit_to_poly,
    list_to_reverse_bit_order,
    get_root_of_unity,
    reverse_bit_order,
    is_power_of_two,
    eval_poly_at,
    get_extended_data
)
from fk20_single_cuda import (
    toeplitz_part1,
    toeplitz_part2,
    toeplitz_part3
)

# FK20 Method to compute all proofs
# Toeplitz multiplication via http://www.netlib.org/utk/people/JackDongarra/etemplates/node384.html
# Multi proof method

def fk20_multi(polynomial, l, setup):
    """
    For a polynomial of size n, let w be a n-th root of unity. Then this method will return
    k=n/l KZG proofs for the points
        proof[0]: w^(0*l + 0), w^(0*l + 1), ... w^(0*l + l - 1)
        proof[1]: w^(1*l + 0), w^(1*l + 1), ... w^(1*l + l - 1)
        ...
        proof[i]: w^(i*l + 0), w^(i*l + 1), ... w^(i*l + l - 1)
        ...
    """

    n = len(polynomial)
    k = n // l
    assert is_power_of_two(n)
    assert is_power_of_two(l)
    assert k >= 1
    
    # Preprocessing part -- this is independent from the polynomial coefficients and can be
    # done before the polynomial is known, it only needs to be computed once
    xext_fft = []
    for i in range(l):
        x = setup[0][n - l - 1 - i::-l] + [b.Z1]	# len(x) == n//l == k
        xext_fft.append(toeplitz_part1(x))		# len(xext_fft) == 2*k

    hext_fft = [b.Z1] * 2 * k
    for i in range(l):

        toeplitz_coefficients = polynomial[- i - 1::l] + [0] * (k + 1) + polynomial[2 * l - i - 1: - l - i:l]

        # Compute the vector h from the paper using a Toeplitz matrix multiplication
        hext_fft = [b.add(v, w) for v, w in zip(hext_fft, toeplitz_part2(toeplitz_coefficients, xext_fft[i]))]
    
    h = toeplitz_part3(hext_fft)

    # The proofs are the DFT of the h vector
    return fft(h, MODULUS, get_root_of_unity(k))


def fk20_multi_data_availability_optimized(polynomial, l, setup):
    """
    FK20 multi-proof method, optimized for dava availability where the top half of polynomial
    coefficients == 0
    """

    assert is_power_of_two(len(polynomial))
    n = len(polynomial) // 2
    k = n // l
    assert is_power_of_two(n)
    assert is_power_of_two(l)
    assert k >= 1

    assert all(x == 0 for x in polynomial[n:])
    reduced_polynomial = polynomial[:n]

    print('__managed__ g1p_t')
    print(f'setup[{len(setup[0])}] = ', end='')
    print('{')
    for i0 in range(len(setup[0])):
        print('\t{ //', i0)
        if setup[0][i0][2].n == 0: # z == 0
            print('\t\t{ 0, 0, 0, 0, 0, 0 },')
            print('\t\t{ 1, 0, 0, 0, 0, 0 },')
            print('\t\t{ 0, 0, 0, 0, 0, 0 },')
        else:
            for i1 in range(3):
                print('\t\t{ ', end='')
                for i2 in range(6):
                    t = setup[0][i0][i1].n
                    t >>= 64*i2
                    t &= 0xffffffffffffffff
                    print(f'0x{t:016x}, ', end='')
                print('}, //', "xyz"[i1])
        print('\t},')
    print('};')

    # Preprocessing part -- this is independent from the polynomial coefficients and can be
    # done before the polynomial is known, it only needs to be computed once
    xext_fft = []
    for i in range(l):
        x = setup[0][n-l-1-i : : -l] + [b.Z1]
        xext_fft.append(toeplitz_part1(x))

    print('__managed__ g1p_t')
    print(f'xext_fft[{l}][{2*k}] = ', end='')
    print('{')
    for i0 in range(l):
        print('\t{ //', i0)
        for i1 in range(2*k):
            print('\t\t{ //', i1)
            if xext_fft[i0][i1][2].n == 0: # z == 0
                print('\t\t\t{ 0, 0, 0, 0, 0, 0 },')
                print('\t\t\t{ 1, 0, 0, 0, 0, 0 },')
                print('\t\t\t{ 0, 0, 0, 0, 0, 0 },')
            else:
                for i2 in range(3):
                    print('\t\t\t{ ', end='')
                    for i3 in range(6):
                        t = xext_fft[i0][i1][i2].n
                        t >>= 64*i3
                        t &= 0xffffffffffffffff
                        print(f'0x{t:016x}, ', end='')
                    print('}, //', "xyz"[i2])
            print('\t\t},')
        print('\t},')
    print('};')

    add_instrumentation()

    hext_fft = [b.Z1] * 2 * k

    print('__managed__ fr_t')
    print(f'toeplitz_coefficients[{l}][{2*k}] = ', end='')
    print('{')

    for i in range(l):

        toeplitz_coefficients = reduced_polynomial[- i - 1::l] + [0] * (k + 1) \
             + reduced_polynomial[2 * l - i - 1: - l - i:l]

        print('\t{')
        for i0 in range(len(toeplitz_coefficients)):
            print('\t\t{ ', end='')
            for i1 in range(4):
                t = toeplitz_coefficients[i0]
                t >>= 64*i1
                t &= 0xffffffffffffffff
                print(f'0x{t:016x}, ', end='')
            print('}, //', i0)
        print('\t},')

    print('};')

    print('__managed__ fr_t')
    print(f'toeplitz_coefficients_fft[{l}][{2*k}] = ', end='')
    print('{')

    for i in range(l):

        toeplitz_coefficients = reduced_polynomial[- i - 1::l] + [0] * (k + 1) \
             + reduced_polynomial[2 * l - i - 1: - l - i:l]

        # Compute the vector h from the paper using a Toeplitz matrix multiplication
        hext_fft = [b.add(v, w) for v, w in zip(hext_fft, toeplitz_part2(toeplitz_coefficients, xext_fft[i]))]

    print('};')

    print('__managed__ g1p_t')
    print(f'hext_fft[{2*k}] = ', end='')
    print('{')
    for i0 in range(2*k):
        print('\t{ //', i0)
        if hext_fft[i0][2].n == 0: # z == 0
            print('\t\t{ 0, 0, 0, 0, 0, 0 },')
            print('\t\t{ 1, 0, 0, 0, 0, 0 },')
            print('\t\t{ 0, 0, 0, 0, 0, 0 },')
        else:
            for i1 in range(3):
                print('\t\t{ ', end='')
                for i2 in range(6):
                    t = hext_fft[i0][i1].n
                    t >>= 64*i2
                    t &= 0xffffffffffffffff
                    print(f'0x{t:016x}, ', end='')
                print('}, //', "xyz"[i1])
        print('\t},')
    print('};')

    # Final FFT done after summing all h vectors
    h = toeplitz_part3(hext_fft)

    print('__managed__ g1p_t')
    print(f'h[{2*k}] = ', end='')   # include the [0]*k appended to h
    print('{')
    for i0 in range(2*k):
        print('\t{ //', i0)
        if (i0 >= 256) or (h[i0][2].n == 0): # z == 0
            print('\t\t{ 0, 0, 0, 0, 0, 0 },')
            print('\t\t{ 1, 0, 0, 0, 0, 0 },')
            print('\t\t{ 0, 0, 0, 0, 0, 0 },')
        else:
            for i1 in range(3):
                print('\t\t{ ', end='')
                for i2 in range(6):
                    t = h[i0][i1].n
                    t >>= 64*i2
                    t &= 0xffffffffffffffff
                    print(f'0x{t:016x}, ', end='')
                print('}, //', "xyz"[i1])
        print('\t},')
    print('};')

    h = h + [b.Z1] * k

    # The proofs are the DFT of the h vector
    h_fft = fft(h, MODULUS, get_root_of_unity(2 * k))

    print('__managed__ g1p_t')
    print(f'h_fft[{2*k}] = ', end='')
    print('{')
    for i0 in range(2*k):
        print('\t{ //', i0)
        if h_fft[i0][2].n == 0: # z == 0
            print('\t\t{ 0, 0, 0, 0, 0, 0 },')
            print('\t\t{ 1, 0, 0, 0, 0, 0 },')
            print('\t\t{ 0, 0, 0, 0, 0, 0 },')
        else:
            for i1 in range(3):
                print('\t\t{ ', end='')
                for i2 in range(6):
                    t = h_fft[i0][i1].n
                    t >>= 64*i2
                    t &= 0xffffffffffffffff
                    print(f'0x{t:016x}, ', end='')
                print('}, //', "xyz"[i1])
        print('\t},')
    print('};')

    return h_fft


def data_availabilty_using_fk20_multi(polynomial, l, setup):
    """
    Computes all the KZG proofs for data availability checks. This involves sampling on the double domain
    and reordering according to reverse bit order
    """
    assert is_power_of_two(len(polynomial))
    n = len(polynomial)
    extended_polynomial = polynomial + [0] * n

    all_proofs = fk20_multi_data_availability_optimized(extended_polynomial, l, setup)

    return list_to_reverse_bit_order(all_proofs)


def add_instrumentation():
    global multiplication_count
    
    multiplication_count = 0

    # Add counter to multiply function for statistics
    b_multiply_ = b.multiply
    def multiply_and_count(*args):
        global multiplication_count
        multiplication_count += 1

        return b_multiply_(*args)

    b.multiply = multiply_and_count


if __name__ == "__main__":
    x = 1
    y = 1
    args = len(sys.argv)
    if args > 1:
        x = int(sys.argv[1])
    if args > 2:
        y = int(sys.argv[2])

#   polynomial = [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 1, 15, MODULUS - 1, 1000, MODULUS - 134, 33] * 256

    polynomial = []

    for i in range(4096//2):
        x %= MODULUS
        y %= MODULUS
        polynomial += [x, y]
        x += y
        y += x

    n = len(polynomial)

    print('__managed__ fr_t')
    print(f'polynomial[{n}] = ', end='')
    print('{')
    for i0 in range(n):
        print('\t{ ', end='')
        for i1 in range(4):
            t = polynomial[i0]
            t >>= 64*i1
            t &= 0xffffffffffffffff
            print(f'0x{t:016x}, ', end='')
        print('}, //', i0)
    print('};')
    setup = generate_setup(1927409816240961209460912649124, n)

    commitment = commit_to_poly(polynomial, setup)

    l = 16
    
    all_proofs = data_availabilty_using_fk20_multi(polynomial, l, setup)
    print('/*')
    print("All KZG proofs computed for data availability (supersampled by factor 2)")
    print("Required {0} G1 multiplications".format(multiplication_count))
    print(n, l, multiplication_count)

    # Now check all positions
    extended_data = get_extended_data(polynomial)

    for pos in range(2 * n // l):
        root_of_unity = get_root_of_unity(n * 2)
        x = pow(root_of_unity, reverse_bit_order(pos, 2 * n // l), MODULUS)
        ys = extended_data[l * pos:l * (pos + 1)]

        subgroup_root_of_unity = get_root_of_unity(l)
        coset = [x * pow(subgroup_root_of_unity, i, MODULUS) for i in range(l)]
        ys2 = [eval_poly_at(polynomial, z) for z in coset]
        assert list_to_reverse_bit_order(ys) == ys2

        assert check_proof_multi(commitment, all_proofs[pos], x, list_to_reverse_bit_order(ys), setup)
        print("Data availability sample check {0} passed".format(pos))
    print('*/')
    
