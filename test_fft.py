#!/usr/bin/env python3

#for data standards, see ../doc/test_fft.md

import FK20Py

import random
import pickle
from tqdm import tqdm

def pointToInt(i : FK20Py.blst.P1) -> int:
    b = i.serialize()
    x=b[:48]
    y=b[48:]
    return int.from_bytes(x, byteorder='big'), int.from_bytes(y, byteorder='big')

def pointToHexString(i : FK20Py.blst.P1) -> str:
    x,y = pointToInt(i)
    return '{:096x}{:096x}'.format(x,y)

MAX_DEGREE_POLY = FK20Py.MODULUS-1
N_POINTS = 512 #Number of points in the Poly
N_TESTS = 2 #Number of tests to generate

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

    for testN in tqdm(range(nTest)):
        poly = genRandonPoly()
        print('polynomial', *[f'{i:0{256//4}x}' for i in poly], sep=' ')
        polys.append(poly)

        fftin, fftout = generateTest(poly, setup)
        print(f"fftTestInput_{testN}",  stringfyFFT_Trace(fftin))
        print(f"fftTestOutput_{testN}", stringfyFFT_Trace(fftout))
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

import sys
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__ == '__main__':
    random.seed(0) #remove after debug
    print(f"NTESTS {N_TESTS}")
    test = generateAllTest(N_TESTS)
    with open("testfft.pickle", 'wb') as f:
        pickle.dump(test, f)

    eprint(len(test['inputs'][0]))

