// bls12_381: Arithmetic for BLS12-381
// Copyright 2022 Dag Arne Osvik

#ifndef G1_CUH
#define G1_CUH

#include <stdint.h>

typedef uint64_t g1a_t[12], g1p_t[18];

extern __device__ void g1a_fromUint64(uint64_t *a, const uint64_t *x, const uint64_t *y);
extern __device__ void g1a_fromFp(uint64_t *a, const uint64_t *x, const uint64_t *y);
extern __device__ void g1a_fromG1p(uint64_t *a, const uint64_t *p);
extern __device__ void g1a_print(const char *s, const uint64_t *a);
extern __device__ void g1a_cpy(uint64_t *a, const uint64_t *b);

extern __device__ void g1p_toUint64(uint64_t *p, uint64_t *x, uint64_t *y, uint64_t *z);
extern __device__ void g1p_fromUint64(uint64_t *p, uint64_t *x, uint64_t *y, uint64_t *z);
__device__ inline void g1p_fromFp(uint64_t *p, uint64_t *x, uint64_t *y, uint64_t *z) {
    g1p_fromUint64(p, x, y, z);
}
extern __device__ void g1p_fromG1a(uint64_t *p, const uint64_t *q); // g1p p; g1a q
extern __device__ void g1p_print(const char *s, const uint64_t *p); // g1p p
extern __device__ void g1p_cpy(uint64_t *p, const uint64_t *q);     // g1p p,q

extern __device__ bool g1p_eq(uint64_t *p, uint64_t *q);            // g1p p,q
extern __device__ bool g1p_neq(uint64_t *p, uint64_t *q);           // g1p p,q
extern __device__ bool g1p_isInf(const uint64_t *p);                // g1p p
extern __device__ bool g1p_isPoint(const uint64_t *p);              // g1p p

extern __device__ void g1p_neg(uint64_t *p);                        // g1p p
extern __device__ void g1p_scale(uint64_t *p, const uint64_t *s);   // g1p p; fp x
extern __device__ void g1p_dbl(uint64_t *p);                        // g1p p
extern __device__ void g1p_add(uint64_t *p, const uint64_t *q);     // g1p p,q
extern __device__ void g1p_sub(uint64_t *p, const uint64_t *q);     // g1p p,q
extern __device__ void g1p_addsub(uint64_t *p, uint64_t *q);        // g1p p,q
extern __device__ void g1p_mul(uint64_t *p, uint64_t *x);           // g1p p; fr x

extern __device__ void g1a_inf(uint64_t *a);                        // g1a a
extern __device__ void g1a_gen(uint64_t *a);                        // g1a a

extern __device__ void g1p_inf(uint64_t *p);                        // g1p p
extern __device__ void g1p_gen(uint64_t *p);                        // g1p p

#endif

// vim: ts=4 et sw=4 si
