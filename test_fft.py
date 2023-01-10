#for data standards, see ../doc/test_fft.md

import FK20Py

import random
import pickle

def pointToInt(i : FK20Py.blst.P1) -> int:
    bytes = i.serialize()
    return int.from_bytes(bytes, byteorder='big')

def pointToHexString(i : FK20Py.blst.P1) -> str:
    integer = pointToInt(i)
    return '{:0192x}'.format(integer)

MAX_DEGREE_POLY = FK20Py.MODULUS-1
N_POINTS = 512 #Number of points in the Poly
N_TESTS = 10 #Number of tests to generate

def stringfyFFT_Trace(fft) -> str:
    return ' ' .join(pointToHexString(point) for point in fft)
    
def genRandonPoly():
    return [random.randint(1, MAX_DEGREE_POLY) for _ in range(N_POINTS)]

def generateTest(polynomial:list, setup:int):
    '''
    generates a test case for the fft part of the algorithm
    '''
    setup = FK20Py.generate_setup(setup, len(polynomial))
    _ = FK20Py.commit_to_poly(polynomial, setup)
    # Computing the proofs on the double 
    fftin, fftout = FK20Py.fftTrace(polynomial, setup)
    return fftin, fftout

def generateAllTest(nTest=1):
    polys=[]
    inputs=[]
    outputs = []
    setup = random.getrandbits(256)
    print(f'setup {setup:0{256//4}x}')

    for testN in range(nTest):
        poly = genRandonPoly()
        print('polynomial', *[f'{i:0{256//4}x}' for i in poly], sep=' ')
        polys.append(poly)

        fftin, fftout = generateTest(poly, setup)
        print(f"fftTestInput_{testN}", stringfyFFT_Trace(fftin))
        inputs.append ([pointToInt(i) for i in fftin])
        outputs.append([pointToInt(i) for i in fftout])
    return {'polys':polys, 
            'inputs':inputs,
            'outputs':outputs, 
            'setup':setup}

def printExpectedOutput(test, skip=True):
    if skip: return
    print("#")
    for idx, output in enumerate(test['outputs']):
        print(f"fftTestOutput_{idx}", *[f'{i:0{764//4}x}' for i in output])


if __name__ == '__main__':
    random.seed(0) #remove after debug
    test = generateAllTest(N_TESTS)
    with open("testfft.pickle", 'wb') as f:
        pickle.dump(test, f)

    printExpectedOutput(test, False)

