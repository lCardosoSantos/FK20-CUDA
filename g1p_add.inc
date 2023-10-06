g1p_add:
    // cal LOAD2
    mov.u32 %fun, LOAD2_;
    mov.u32 %ret, retg1p_add_0_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_0:
// BLOCK 0
	// fp_add(ta, X1, Y1); // t4
	FP_MOV(b, x)
	FP_MOV(c, y)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_1_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_1:
	FP_MOV(ta, a)

	// fp_add(tb, Y1, Z1); // t8
	FP_MOV(b, y)
	FP_MOV(c, z)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_2_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_2:
	FP_MOV(tb, a)

	// fp_add(tc, Z1, X1); // t4
	FP_MOV(b, z)
	FP_MOV(c, x)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_3_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_3:
	FP_MOV(tc, a)

// BLOCK 1
	// fp_mul(X1, X1, X2); // t1
	FP_MOV(b, x)
	FP_MOV(c, u)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_4_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_4:
	FP_MOV(x, a) 

	// fp_mul(Y1, Y1, Y2); // t2
	FP_MOV(b, y)
	FP_MOV(c, v)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_5_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_5:
	FP_MOV(y, a) 

	// fp_mul(Z1, Z1, Z2); // t3
	FP_MOV(b, z)
	FP_MOV(c, w)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_6_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_6:
	FP_MOV(z, a) 

// BLOCK 2
	// fp_add(td, X2, Y2); // t4
	FP_MOV(b, u)
	FP_MOV(c, v)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_7_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_7:
	FP_MOV(td, a)

	// fp_add(Y2, Y2, Z2); // t9
	FP_MOV(b, v)
	FP_MOV(c, w)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_8_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_8:
	FP_MOV(v, a)

	// fp_add(Z2, Z2, X2); // te
	FP_MOV(b, w)
	FP_MOV(c, u)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_9_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_9:
	FP_MOV(w, a)

// BLOCK 3
	// fp_mul(X2, td, ta); // t5
	FP_MOV(b, td)
	FP_MOV(c, ta)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_10_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_10:
	FP_MOV(u, a) 

	// fp_mul(Y2, Y2, tb); // t1
	FP_MOV(b, v)
	FP_MOV(c, tb)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_11_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_11:
	FP_MOV(v, a) 

	// fp_mul(Z2, Z2, tc); // tf
	FP_MOV(b, w)
	FP_MOV(c, tc)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_12_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_12:
	FP_MOV(w, a) 

// BLOCK 4
	// fp_x3(ta, X1);      // ti
	FP_MOV(a, x)
    // cal fp_x3
    mov.u32 %fun, fp_x3_;
    mov.u32 %ret, retg1p_add_13_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_13:
	FP_MOV(ta, a)

	// fp_add(tb, Y1, Z1); // t2
	FP_MOV(b, y)
	FP_MOV(c, z)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_14_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_14:
	FP_MOV(tb, a)

	// fp_add(tc, Z1, X1); // tg
	FP_MOV(b, z)
	FP_MOV(c, x)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_15_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_15:
	FP_MOV(tc, a)

	// fp_x12(td, Z1);     // tk
	FP_MOV(a, z)
    // cal fp_x12
    mov.u32 %fun, fp_x12_;
    mov.u32 %ret, retg1p_add_16_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_16:
	FP_MOV(td, a)

// BLOCK 5
	// fp_add(X1, X1, Y1); // t6
	FP_MOV(b, x)
	FP_MOV(c, y)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_17_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_17:
	FP_MOV(x, a)

	// fp_add(Z1, Y1, td); // tl
	FP_MOV(b, y)
	FP_MOV(c, td)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_18_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_18:
	FP_MOV(z, a)

	// fp_sub(Y1, Y1, td); // tm
	FP_MOV(b, y)
	FP_MOV(c, td)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_add_19_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_19:
	FP_MOV(y, a)

// BLOCK 6
	// fp_sub(X1, X2, X1); // t7
	FP_MOV(b, u)
	FP_MOV(c, x)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_add_20_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_20:
	FP_MOV(x, a)

	// fp_mul(X2, X1, ta); // ts
	FP_MOV(b, x)
	FP_MOV(c, ta)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_21_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_21:
	FP_MOV(u, a) 

// BLOCK 7
	// fp_mul(X1, X1, Y1); // tp
	FP_MOV(b, x)
	FP_MOV(c, y)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_22_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_22:
	FP_MOV(x, a) 

	// fp_mul(Y1, Y1, Z1); // tr
	FP_MOV(b, y)
	FP_MOV(c, z)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_23_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_23:
	FP_MOV(y, a) 

// BLOCK 8
	// fp_sub(Y2, Y2, tb); // t3
	FP_MOV(b, v)
	FP_MOV(c, tb)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_add_24_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_24:
	FP_MOV(v, a)

	// fp_mul(Z1, Z1, Y2); // tt
	FP_MOV(b, z)
	FP_MOV(c, v)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_25_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_25:
	FP_MOV(z, a) 

	// fp_sub(Z2, Z2, tc); // th
	FP_MOV(b, w)
	FP_MOV(c, tc)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_add_26_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_26:
	FP_MOV(w, a)

// BLOCK 9
	// fp_x12(Z2, Z2);     // tn
	FP_MOV(a, w)
    // cal fp_x12
    mov.u32 %fun, fp_x12_;
    mov.u32 %ret, retg1p_add_27_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_27:
	FP_MOV(w, a)

	// fp_mul(Y2, Y2, Z2); // to
	FP_MOV(b, v)
	FP_MOV(c, w)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_28_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_28:
	FP_MOV(v, a) 

	// fp_mul(Z2, Z2, ta); // tq
	FP_MOV(b, w)
	FP_MOV(c, ta)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_add_29_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_29:
	FP_MOV(w, a) 

// BLOCK 10
	// fp_sub(X1, X1, Y2); // X3
	FP_MOV(b, x)
	FP_MOV(c, v)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_add_30_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_30:
	FP_MOV(x, a)

	// fp_add(Y1, Y1, Z2); // Y3
	FP_MOV(b, y)
	FP_MOV(c, w)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_31_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_31:
	FP_MOV(y, a)

	// fp_add(Z1, Z1, X2); // Z3
	FP_MOV(b, z)
	FP_MOV(c, u)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_add_32_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_32:
	FP_MOV(z, a)

    // cal STORE1
    mov.u32 %fun, STORE1_;
    mov.u32 %ret, retg1p_add_33_;
    brx.idx.uni %fun, fp_fun;
retg1p_add_33:
	ret.uni;

