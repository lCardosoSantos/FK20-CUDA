'''
operates over objects in py_ecc.optimized_bls12_381 to represent them as packed structs.

The C structs are the following:
```
typedef uint64_t fr_t[4];
typedef uint64_t fp_t[6];

typedef struct {
    fp_t x, y, z;
} g1p_t;
```
Also stores the globals used to dump data

'''
from py_ecc import optimized_bls12_381 as b
import struct
mask = 0xffffffffffffffff

g_polynomial = None
g_setup = None
g_xext_fft = None
g_toeplitz_coefficients = []
g_toeplitz_coefficients_fft = []
g_hext_fft = None
g_h = None
g_h_fft = None


def frToByte(p: int):
    return struct.pack('QQQQ', p&mask, p>>64&mask, p>>(64*2)&mask, p>>(64*3)&mask)

def fpToByte(v: int):
    return struct.pack('6Q', v >>(64*0)&mask, 
                             v >>(64*1)&mask, 
                             v >>(64*2)&mask, 
                             v >>(64*3)&mask, 
                             v >>(64*4)&mask, 
                             v >>(64*5)&mask)

def g1ToByte(g1: tuple):
    #todo: check for zero
    if b.is_inf(g1):
        #dags representation of zero
        return struct.pack('18Q', 0, 0, 0, 0, 0, 0,
                                  1, 0, 0, 0, 0, 0, 
                                  0, 0, 0, 0, 0, 0)
    else:
        return struct.pack('18Q', g1[0].n>>(64*0)&mask, g1[0].n>>(64*1)&mask, g1[0].n>>(64*2)&mask, g1[0].n>>(64*3)&mask, g1[0].n>>(64*4)&mask, g1[0].n>>(64*5)&mask, 
                                  g1[1].n>>(64*0)&mask, g1[1].n>>(64*1)&mask, g1[1].n>>(64*2)&mask, g1[1].n>>(64*3)&mask, g1[1].n>>(64*4)&mask, g1[1].n>>(64*5)&mask,
                                  g1[2].n>>(64*0)&mask, g1[2].n>>(64*1)&mask, g1[2].n>>(64*2)&mask, g1[2].n>>(64*3)&mask, g1[2].n>>(64*4)&mask, g1[2].n>>(64*5)&mask,)
