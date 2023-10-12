

Prototype:

`g1p_ptx_fun(int op, g1p_t* out1, g1p_t* out0, g1p_t const* in1, g1p_t const* in0);`

Registers: 
'''
a   -> accumulator
op1, op2 -> operand
x1, y1, z1, z2, y2, z2 -> fp[6]-sized
t0, t1, t2, t3 -> 64biy
'''


`<Label>` is replaced with

```
    move.u32 %fun label;
    move.u32 %ret retLabel;
    brx.idx.uni %fun, fp_fun;
retLabel:
```

Probably cannot make a C Macro with it, unless we can use `__COUNTER__`

```C++
#define MOVFP(a, b)   \
        mov.u32 a##0, b##0;\
        mov.u32 a##1, b##1;\
        mov.u32 a##2, b##2;\
        mov.u32 a##3, b##3;\
        mov.u32 a##4, b##4;\
        mov.u32 a##5, b##5

g1p_add:
    <LOAD2> 
    // fp_add(t0, x1, y1); // t3
    MOVFP op1, x1;
    MOVFP op2, y1;
    <fp_add>       //result in acc
    MOVFP t0, a;  //move from acc to tmp

    // fp_add(t1, y1, z1); // t8
    MOVFP op1, y1;    //HERE DAG!
    MOVFP op2, z1;
    <fp_add>       //result in acc
    MOVFP t1, a;   //move from acc to tmp

    // fp_add(t2, z1, x1); // td
    MOVFP op1, z1;
    MOVFP op2, x1;
    <fp_add>       //result in acc
    MOVFP t2, a;  //move from acc to tmp

    // fp_mul(x1, x1, x2); // t0
    MOVFP op1, x1;
    MOVFP op2, x2;
    <fp_mul>       //result in acc
    MOVFP x1, a;  //move from acc to tmp

    // fp_mul(y1, y1, y2); // t1
    MOVFP op1, y1;
    MOVFP op2, y2;
    <fp_mul>       //result in acc
    MOVFP y1, a;  //move from acc to tmp

    // fp_mul(z1, z1, z2); // t2
    MOVFP op1, z1;
    MOVFP op2, z2;
    <fp_mul>       //result in acc
    MOVFP z1, a;  //move from acc to tmp

    // fp_add(t3, x2, y2); // t4
    MOVFP op1, x2;
    MOVFP op2, y2;
    <fp_add>       //result in acc
    MOVFP t3, a;  //move from acc to tmp

    // fp_add(y2, y2, z2); // t9
    MOVFP op1, y2;
    MOVFP op2, z2;
    <fp_add>       //result in acc
    MOVFP y2, a;  //move from acc to tmp

    // fp_add(z2, z2, x2); // te
    MOVFP op1, z2;
    MOVFP op2, x2;
    <fp_add>       //result in acc
    MOVFP z2, a;  //move from acc to tmp

    // fp_mul(x2, t3, t0); // t5
    MOVFP op1, t3;
    MOVFP op2, t0;
    <fp_mul>       //result in acc
    MOVFP x2, a;  //move from acc to tmp

    // fp_mul(y2, y2, t1); // ta
    MOVFP op1, y2;
    MOVFP op2, t1;
    <fp_mul>       //result in acc
    MOVFP y2, a;  //move from acc to tmp

    // fp_mul(z2, z2, t2); // tf
    MOVFP op1, z2;
    MOVFP op2, t2;
    <fp_mul>       //result in acc
    MOVFP z2, a;  //move from acc to tmp

    // fp_x3(t0, x1);      // ti
    MOVFP op1, x1;
    <fp_x3>       //result in acc
    MOVFP t0, a;  //move from acc to tmp

    // fp_add(t1, y1, z1); // tb
    MOVFP op1, y1;
    MOVFP op2, z1;
    <fp_add>       //result in acc
    MOVFP t1, a;  //move from acc to tmp

    // fp_add(t2, z1, x1); // tg
    MOVFP op1, z1;
    MOVFP op2, x1;
    <fp_add>       //result in acc
    MOVFP t2, a;  //move from acc to tmp

    // fp_x12(t3, z1);     // tk
    MOVFP op1, z1;
    <fp_x12>       //result in acc
    MOVFP t3, a;  //move from acc to tmp

    // fp_add(x1, x1, y1); // t6
    MOVFP op1, x1;
    MOVFP op2, y1;
    <fp_add>       //result in acc
    MOVFP x1, a;  //move from acc to tmp

    // fp_add(z1, y1, t3); // tl
    MOVFP op1, y1;
    MOVFP op2, t3;
    <fp_add>       //result in acc
    MOVFP z1, a;  //move from acc to tmp

    // fp_sub(y1, y1, t3); // tm
    MOVFP op1, y1;
    MOVFP op2, t3;
    <fp_sub>       //result in acc
    MOVFP y1, a;  //move from acc to tmp

    // fp_sub(x1, x2, x1); // t7
    MOVFP op1, x2;
    MOVFP op2, x1;
    <fp_sub>       //result in acc
    MOVFP x1, a;  //move from acc to tmp

    // fp_mul(x2, x1, t0); // ts
    MOVFP op1, x1;
    MOVFP op2, t0;
    <fp_mul>       //result in acc
    MOVFP x2, a;  //move from acc to tmp

    // fp_mul(x1, x1, y1); // tp
    MOVFP op1, x1;
    MOVFP op2, y1;
    <fp_mul>       //result in acc
    MOVFP x1, a;  //move from acc to tmp

    // fp_mul(y1, y1, z1); // tr
    MOVFP op1, y1;
    MOVFP op2, z1;
    <fp_mul>       //result in acc
    MOVFP y1, a;  //move from acc to tmp

    // fp_sub(y2, y2, t1); // tc
    MOVFP op1, y2;
    MOVFP op2, t1;
    <fp_sub>       //result in acc
    MOVFP y2, a;  //move from acc to tmp

    // fp_mul(z1, z1, y2); // tt
    MOVFP op1, z1;
    MOVFP op2, y2;
    <fp_mul>       //result in acc
    MOVFP z1, a;  //move from acc to tmp

    // fp_sub(z2, z2, t2); // th
    MOVFP op1, z2;
    MOVFP op2, t2;
    <fp_sub>       //result in acc
    MOVFP z2, a;  //move from acc to tmp

    // fp_x12(z2, z2);     // tn
    MOVFP op1, z2;
    <fp_x12>       //result in acc
    MOVFP z2, a;  //move from acc to tmp

    // fp_mul(y2, y2, z2); // to
    MOVFP op1, y2;
    MOVFP op2, z2;
    <fp_mul>       //result in acc
    MOVFP y2, a;  //move from acc to tmp

    // fp_mul(z2, z2, t0); // tq
    MOVFP op1, z2;
    MOVFP op2, t0;
    <fp_mul>       //result in acc
    MOVFP z2, a;  //move from acc to tmp

    // fp_sub(x1, x1, y2); // x3
    MOVFP op1, x1;
    MOVFP op2, y2;
    <fp_sub>       //result in acc
    MOVFP x1, a;  //move from acc to tmp

    // fp_add(y1, y1, z2); // y3
    MOVFP op1, y1;
    MOVFP op2, z2;
    <fp_add>       //result in acc
    MOVFP y1, a;  //move from acc to tmp

    // fp_add(z1, z1, x2); // z3
    MOVFP op1, z1;
    MOVFP op2, x2;
    <fp_add>       //result in acc
    MOVFP z1, a;  //move from acc to tmp

    //store outputs
    ret;

LOAD2:
    load (u, v, w) in1
LOAD1:
    load(x, y, z) in0
    jumpback

```

Load extends into:
QUESTION: Can the parameterized register names have more than one character in the prefix?

```C++
LOAD2:
    ld.u64 	x20, [parIn0+0];
    ld.u64 	x21, [parIn0+8];
    ld.u64 	x22, [parIn0+16];
    ld.u64 	x23, [parIn0+24];
    ld.u64 	x24, [parIn0+32];
    ld.u64 	x25, [parIn0+40];
    ld.u64 	y26, [parIn0+48];
    ld.u64 	y27, [parIn0+56];
    ld.u64 	y28, [parIn0+64];
    ld.u64 	y29, [parIn0+72];
    ld.u64 	y210, [parIn0+80];
    ld.u64 	y211, [parIn0+88];
    ld.u64 	z212, [parIn0+96];
    ld.u64 	z213, [parIn0+104];
    ld.u64 	z214, [parIn0+112];
    ld.u64 	z215, [parIn0+120];
    ld.u64 	z216, [parIn0+128];
    ld.u64 	z217, [parIn0+136];
LOAD1:
    ld.u64 	x10, [parIn0+0];
    ld.u64 	x11, [parIn0+8];
    ld.u64 	x12, [parIn0+16];
    ld.u64 	x13, [parIn0+24];
    ld.u64 	x14, [parIn0+32];
    ld.u64 	x15, [parIn0+40];
    ld.u64 	y16, [parIn0+48];
    ld.u64 	y17, [parIn0+56];
    ld.u64 	y18, [parIn0+64];
    ld.u64 	y19, [parIn0+72];
    ld.u64 	y110, [parIn0+80];
    ld.u64 	y111, [parIn0+88];
    ld.u64 	z112, [parIn0+96];
    ld.u64 	z113, [parIn0+104];
    ld.u64 	z114, [parIn0+112];
    ld.u64 	z115, [parIn0+120];
    ld.u64 	z116, [parIn0+128];
    ld.u64 	z117, [parIn0+136];
    jumpback

```
