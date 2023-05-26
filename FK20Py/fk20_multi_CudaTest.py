from py_ecc import optimized_bls12_381 as b
from fft import fft
import kzg_proofs
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
from fk20_single import (
    toeplitz_part1,
    toeplitz_part2,
    toeplitz_part3
)
import toByte


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

    toByte.g_setup = setup[0]

    # Preprocessing part -- this is independent from the polynomial coefficients and can be
    # done before the polynomial is known, it only needs to be computed once
    xext_fft = []
    for i in range(l):
        x = setup[0][n-l-1-i : : -l] + [b.Z1]
        xext_fft.append(toeplitz_part1(x))

    toByte.g_xext_fft = xext_fft

    #add_instrumentation()

    hext_fft = [b.Z1] * 2 * k

    for i in range(l):

        toeplitz_coefficients = reduced_polynomial[- i - 1::l] + [0] * (k + 1) \
             + reduced_polynomial[2 * l - i - 1: - l - i:l]

        toByte.g_toeplitz_coefficients.append(toeplitz_coefficients)

    for i in range(l):

        toeplitz_coefficients = reduced_polynomial[- i - 1::l] + [0] * (k + 1) \
             + reduced_polynomial[2 * l - i - 1: - l - i:l]

        # Compute the vector h from the paper using a Toeplitz matrix multiplication
        hext_fft = [b.add(v, w) for v, w in zip(hext_fft, toeplitz_part2(toeplitz_coefficients, xext_fft[i]))]


    toByte.g_hext_fft = hext_fft

    # Final FFT done after summing all h vectors
    h = toeplitz_part3(hext_fft)

    h = h + [b.Z1] * k

    toByte.g_h = h

    # The proofs are the DFT of the h vector
    h_fft = fft(h, MODULUS, get_root_of_unity(2 * k))

    toByte.g_h_fft = h_fft

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


def genCanonical():
    '''generate the cannonical testcase'''
    polynomial = [1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 1, 15, MODULUS - 1, 1000, MODULUS - 134, 33] * 256
    n = len(polynomial)
    toByte.g_polynomial = polynomial
    setup = generate_setup(1927409816240961209460912649124, n)
    commitment = commit_to_poly(polynomial, setup)
    l = 16
    all_proofs = data_availabilty_using_fk20_multi(polynomial, l, setup)

def genRandom():
    '''generate random testCase'''
    import secrets #uses the most secure source of randomness available in the system.
    n = 4096 #Lenght of the polynomial
    polynomial = [secrets.randbelow(MODULUS) for _ in range(n)]
    toByte.g_polynomial = polynomial
    setup = generate_setup(secrets.randbits(256), n)
    commitment = commit_to_poly(polynomial, setup)
    l = 16
    all_proofs = data_availabilty_using_fk20_multi(polynomial, l, setup)

def dumpData():
    '''Dump the data stored in the globals to STDIO'''
    import sys
    def dump(l):
        bc=0
        for i in l:
            sys.stdout.buffer.write(i)
            bc+=len(i)
        return bc

    #fr_t polynomial[4096]
    packedData=[]
    for p in toByte.g_polynomial:
        packedData.append(toByte.frToByte(p))
    bc = dump(packedData)
    print(f'written {bc} bytes for polynomial', file=sys.stderr)

    #g1p_t setup[4097]
    packedData=[]
    for g1 in toByte.g_setup:
        packedData.append(toByte.g1ToByte(g1))
    bc = dump(packedData)
    print(f'written {bc} bytes for setup', file=sys.stderr)

    #g1p_t xext_fft[16][512]
    packedData=[]
    for x in toByte.g_xext_fft:
        for g1 in x:
            packedData.append(toByte.g1ToByte(g1))
    bc = dump(packedData)
    print(f'written {bc} bytes for xext_fft', file=sys.stderr)

    #g1p_t toeplitz_coefficients[16][512]
    packedData=[]
    for x in toByte.g_toeplitz_coefficients:
        for g1 in x:
            packedData.append(toByte.frToByte(g1))
    bc = dump(packedData)
    print(f'written {bc} bytes for toeplitz_coefficients', file=sys.stderr)

    #g1p_t toeplitz_coefficients_fft[16][512]
    packedData=[]
    for x in toByte.g_toeplitz_coefficients_fft:
        for g1 in x:
            packedData.append(toByte.frToByte(g1))
    bc = dump(packedData)
    print(f'written {bc} bytes for g_toeplitz_coefficients_fft', file=sys.stderr)

    #g1p_t hext_fft[512]
    packedData=[]
    for g1 in toByte.g_hext_fft:
        packedData.append(toByte.g1ToByte(g1))
    bc = dump(packedData)
    print(f'written {bc} bytes for hext_fft', file=sys.stderr)

    #g1p_t h[512]
    packedData=[]
    for g1 in toByte.g_h:
        packedData.append(toByte.g1ToByte(g1))
    bc = dump(packedData)
    print(f'written {bc} bytes for h', file=sys.stderr)

    #g1p_t h_fft[512]
    packedData=[]
    for g1 in toByte.g_h_fft:
        packedData.append(toByte.g1ToByte(g1))
    bc = dump(packedData)
    print(f'written {bc} bytes for h_fft', file=sys.stderr)
    
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        genCanonical()
    else:
        genRandom()
    dumpData()
    