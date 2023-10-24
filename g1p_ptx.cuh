#ifndef G1P_PTX_CUH
#define G1P_PTX_CUH

#define OP_ADD 0
//#define OP_SUB 
#define OP_DBL 1
#define OP_ADDSUB 2

extern __device__ void g1m(unsigned long op, g1p_t &out1, g1p_t &out0, g1p_t &in1, g1p_t &in0);

#endif //G1P_PTX_CUH
