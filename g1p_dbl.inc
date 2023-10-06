g1p_dbl:
    // cal LOAD2
    mov.u32 %fun, LOAD2_;
    mov.u32 %ret, retg1p_dbl_0_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_0:
// BLOCK 0
	// fp_mul(X1, X1, Y1);
	FP_MOV(b, x)
	FP_MOV(c, y)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_dbl_1_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_1:
	FP_MOV(x, a) 

	// fp_sqr(ta, Z1);
	FP_MOV(b, z)
    // cal fp_sqr
    mov.u32 %fun, fp_sqr_;
    mov.u32 %ret, retg1p_dbl_2_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_2:
	FP_MOV(ta, a)

	// fp_x12(ta, ta);
	FP_MOV(a, ta)
    // cal fp_x12
    mov.u32 %fun, fp_x12_;
    mov.u32 %ret, retg1p_dbl_3_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_3:
	FP_MOV(ta, a)

	// fp_mul(Z1, Z1, Y1);
	FP_MOV(b, z)
	FP_MOV(c, y)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_dbl_4_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_4:
	FP_MOV(z, a) 

	// fp_sqr(Y1, Y1);
	FP_MOV(b, y)
    // cal fp_sqr
    mov.u32 %fun, fp_sqr_;
    mov.u32 %ret, retg1p_dbl_5_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_5:
	FP_MOV(y, a)

	// fp_x3(tb, ta);
	FP_MOV(a, ta)
    // cal fp_x3
    mov.u32 %fun, fp_x3_;
    mov.u32 %ret, retg1p_dbl_6_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_6:
	FP_MOV(tb, a)

	// fp_sub(tb, Y1, tb);
	FP_MOV(b, y)
	FP_MOV(c, tb)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_dbl_7_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_7:
	FP_MOV(tb, a)

	// fp_mul(X1, X1, tb);
	FP_MOV(b, x)
	FP_MOV(c, tb)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_dbl_8_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_8:
	FP_MOV(x, a) 

	// fp_add(Y1, Y1, ta);
	FP_MOV(b, y)
	FP_MOV(c, ta)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_dbl_9_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_9:
	FP_MOV(y, a)

	// fp_mul(tb, tb, Y1);
	FP_MOV(b, tb)
	FP_MOV(c, y)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_dbl_10_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_10:
	FP_MOV(tb, a) 

	// fp_sub(Y1, Y1, ta);
	FP_MOV(b, y)
	FP_MOV(c, ta)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_dbl_11_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_11:
	FP_MOV(y, a)

	// fp_x8(Y1, Y1);
	FP_MOV(a, y)
    // cal fp_x8
    mov.u32 %fun, fp_x8_;
    mov.u32 %ret, retg1p_dbl_12_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_12:
	FP_MOV(y, a)

	// fp_x2(X1, X1);
	FP_MOV(a, x)
    // cal fp_x2
    mov.u32 %fun, fp_x2_;
    mov.u32 %ret, retg1p_dbl_13_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_13:
	FP_MOV(x, a)

	// fp_mul(Z1, Z1, Y1);
	FP_MOV(b, z)
	FP_MOV(c, y)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_dbl_14_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_14:
	FP_MOV(z, a) 

	// fp_mul(Y1, Y1, ta);
	FP_MOV(b, y)
	FP_MOV(c, ta)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_dbl_15_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_15:
	FP_MOV(y, a) 

	// fp_add(Y1, Y1, tb);
	FP_MOV(b, y)
	FP_MOV(c, tb)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_dbl_16_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_16:
	FP_MOV(y, a)

    // cal STORE1
    mov.u32 %fun, STORE1_;
    mov.u32 %ret, retg1p_dbl_17_;
    brx.idx.uni %fun, fp_fun;
retg1p_dbl_17:
	ret.uni;
