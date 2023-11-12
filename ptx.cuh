// bls12_381: Arithmetic for BLS12-381
// Copyright 2022-2023 Dag Arne Osvik
// Copyright 2022-2023 Luan Cardoso dos Santos

#ifndef PTX_CUH
#define PTX_CUH

#include <cstdint>

// 64 <- 32:32
__device__ __forceinline__ void pack(uint64_t &d, const uint32_t &al, const uint32_t &ah)
{ asm volatile ( "\n\tmov.b64 %0, {%1, %2};" : "=l"(d) : "r"(al), "r"(ah) ); }

// 32:32 <- 64
__device__ __forceinline__ void unpack(uint32_t &dl, uint32_t &dh, const uint64_t &a)
{ asm volatile ( "\n\tmov.b64 {%0, %1}, %2;" : "=r"(dl), "=r"(dh) : "l"(a) ); }


__device__ __forceinline__ void add_cc_u32(uint32_t &d, const uint32_t &a, const uint32_t &b)
{ asm volatile ( "\n\tadd.cc.u32    %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b) ); }

__device__ __forceinline__ void addc_cc_u32(uint32_t &d, const uint32_t &a, const uint32_t &b)
{ asm volatile ( "\n\taddc.cc.u32   %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b) ); }

__device__ __forceinline__ void addc_u32(uint32_t &d, const uint32_t &a, const uint32_t &b)
{ asm volatile ( "\n\taddc.u32      %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b) ); }


__device__ __forceinline__ void sub_cc_u32(uint32_t &d, const uint32_t &a, const uint32_t &b)
{ asm volatile ( "\n\tsub.cc.u32    %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b) ); }

__device__ __forceinline__ void subc_cc_u32(uint32_t &d, const uint32_t &a, const uint32_t &b)
{ asm volatile ( "\n\tsubc.cc.u32   %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b) ); }

__device__ __forceinline__ void subc_u32(uint32_t &d, const uint32_t &a, const uint32_t &b)
{ asm volatile ( "\n\tsubc.u32      %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b) ); }


__device__ __forceinline__ void add_cc_u64(uint64_t &d, const uint64_t &a, const uint64_t &b)
{ asm volatile ( "\n\tadd.cc.u64    %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b) ); }

__device__ __forceinline__ void addc_cc_u64(uint64_t &d, const uint64_t &a, const uint64_t &b)
{ asm volatile ( "\n\taddc.cc.u64   %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b) ); }

__device__ __forceinline__ void addc_u64(uint64_t &d, const uint64_t &a, const uint64_t &b)
{ asm volatile ( "\n\taddc.u64      %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b) ); }


__device__ __forceinline__ void sub_cc_u64(uint64_t &d, const uint64_t &a, const uint64_t &b)
{ asm volatile ( "\n\tsub.cc.u64    %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b) ); }

__device__ __forceinline__ void subc_cc_u64(uint64_t &d, const uint64_t &a, const uint64_t &b)
{ asm volatile ( "\n\tsubc.cc.u64   %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b) ); }

__device__ __forceinline__ void subc_u64(uint64_t &d, const uint64_t &a, const uint64_t &b)
{ asm volatile ( "\n\tsubc.u64      %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b) ); }


__device__ __forceinline__ void mul_wide_u32(uint64_t &d, const uint32_t &a, const uint32_t &b)
{ asm volatile ( "\n\tmul.wide.u32  %0, %1, %2;" : "=l"(d) : "r"(a), "r"(b) ); }

__device__ __forceinline__ void mad_wide_u32(uint64_t &d, const uint32_t &a, const uint32_t &b, const uint64_t &c)
{ asm volatile ( "\n\tmad.wide.u32  %0, %1, %2, %3;" : "=l"(d) : "r"(a), "r"(b), "l"(c) ); }

__device__ __forceinline__ void mad_wide_cc_u32(uint64_t &d, const uint32_t &a, const uint32_t &b, const uint64_t &c)
{
    asm volatile (
        "\n\t{"
        "\n\t .reg.u64 u64;"
        "\n\t mul.wide.u32 u64, %1, %2;"
        "\n\t add.cc.u64   %0, u64, %3;"
        "\n\t}"
        : "=l"(d) : "r"(a), "r"(b), "l"(c)
    );
}

__device__ __forceinline__ void madc_wide_cc_u32(uint64_t &d, const uint32_t &a, const uint32_t &b, const uint64_t &c)
{
    asm volatile (
        "\n\t{"
        "\n\t .reg.u64 u64;"
        "\n\t mul.wide.u32 u64, %1, %2;"
        "\n\t addc.cc.u64  %0, u64, %3;"
        "\n\t}"
        : "=l"(d) : "r"(a), "r"(b), "l"(c)
    );
}

__device__ __forceinline__ void madc_wide_u32(uint64_t &d, const uint32_t &a, const uint32_t &b, const uint64_t &c)
{
    asm volatile (
        "\n\t{"
        "\n\t .reg.u64 u64;"
        "\n\t mul.wide.u32 u64, %1, %2;"
        "\n\t addc.u64     %0, u64, %3;"
        "\n\t}"
        : "=l"(d) : "r"(a), "r"(b), "l"(c)
    );
}

#endif

// vim: ts=4 et sw=4 si
