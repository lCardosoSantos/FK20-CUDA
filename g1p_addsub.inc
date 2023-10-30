g1p_addsub:
    // cal LOAD2
    mov.u32 %fun, LOAD2_;
    mov.u32 %ret, retg1p_addsub_0_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_0:
// BLOCK 0
	// fp_mul(ta, X1, X2); // t1
	FP_MOV(b, x)
	FP_MOV(c, u)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_1_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_1:
	FP_MOV(ta, a) 

	// fp_add(td, X1, Y1); // t4
	FP_MOV(b, x)
	FP_MOV(c, y)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_2_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_2:
	FP_MOV(td, a)

	// fp_add(X1, X1, Z1); // t4
	FP_MOV(b, x)
	FP_MOV(c, z)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_3_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_3:
	FP_MOV(x, a)

	// fp_add(tc, X2, Z2); // te
	FP_MOV(b, u)
	FP_MOV(c, w)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_4_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_4:
	FP_MOV(tc, a)

	// fp_mul(X1, X1, tc); // tf
	FP_MOV(b, x)
	FP_MOV(c, tc)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_5_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_5:
	FP_MOV(x, a) 

// BLOCK 1
	// fp_add(tb, Y2, Z2); // t9
	FP_MOV(b, v)
	FP_MOV(c, w)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_6_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_6:
	FP_MOV(tb, a)

	// fp_mul(tc, Z1, Z2); // t3
	FP_MOV(b, z)
	FP_MOV(c, w)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_7_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_7:
	FP_MOV(tc, a) 

	// fp_add(Z1, Z1, Y1); // t8
	FP_MOV(b, z)
	FP_MOV(c, y)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_8_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_8:
	FP_MOV(z, a)

	// fp_sub(Z2, Z2, Y2); // T9
	FP_MOV(b, w)
	FP_MOV(c, v)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_9_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_9:
	FP_MOV(w, a)

	// fp_mul(Z2, Z2, Z1); // t1
	FP_MOV(b, w)
	FP_MOV(c, z)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_10_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_10:
	FP_MOV(w, a) 

	// fp_mul(Z1, Z1, tb); // t1
	FP_MOV(b, z)
	FP_MOV(c, tb)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_11_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_11:
	FP_MOV(z, a) 

	// fp_sub(Z1, Z1, tc); // t3
	FP_MOV(b, z)
	FP_MOV(c, tc)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_12_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_12:
	FP_MOV(z, a)

	// fp_add(tb, X2, Y2); // t4
	FP_MOV(b, u)
	FP_MOV(c, v)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_13_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_13:
	FP_MOV(tb, a)

	// fp_mul(tb, tb, td); // t5
	FP_MOV(b, tb)
	FP_MOV(c, td)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_14_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_14:
	FP_MOV(tb, a) 

	// fp_sub(X1, X1, tc); // (th)
	FP_MOV(b, x)
	FP_MOV(c, tc)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_15_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_15:
	FP_MOV(x, a)

	// fp_sub(X2, X2, Y2); // T4
	FP_MOV(b, u)
	FP_MOV(c, v)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_16_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_16:
	FP_MOV(u, a)

	// fp_mul(X2, X2, td); // T5
	FP_MOV(b, u)
	FP_MOV(c, td)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_17_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_17:
	FP_MOV(u, a) 

	// fp_sub(X2, X2, ta); // T7
	FP_MOV(b, u)
	FP_MOV(c, ta)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_18_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_18:
	FP_MOV(u, a)

	// fp_mul(Y1, Y1, Y2); // t2
	FP_MOV(b, y)
	FP_MOV(c, v)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_19_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_19:
	FP_MOV(y, a) 

	// fp_x12(Y2, tc);     // tk
	FP_MOV(a, tc)
    // cal fp_x12
    mov.u32 %fun, fp_x12_;
    mov.u32 %ret, retg1p_addsub_20_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_20:
	FP_MOV(v, a)

	// fp_add(Z2, Z2, Y1); // t3
	FP_MOV(b, w)
	FP_MOV(c, y)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_21_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_21:
	FP_MOV(w, a)

	// fp_sub(Z2, Z2, tc); // t3
	FP_MOV(b, w)
	FP_MOV(c, tc)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_22_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_22:
	FP_MOV(w, a)

	// fp_sub(tc, tb, ta); // (t7)
	FP_MOV(b, tb)
	FP_MOV(c, ta)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_23_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_23:
	FP_MOV(tc, a)

	// fp_sub(tc, tc, Y1); // t7
	FP_MOV(b, tc)
	FP_MOV(c, y)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_24_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_24:
	FP_MOV(tc, a)

	// fp_add(X2, X2, Y1); // T7
	FP_MOV(b, u)
	FP_MOV(c, y)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_25_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_25:
	FP_MOV(u, a)

	// fp_sub(Z1, Z1, Y1); // t3
	FP_MOV(b, z)
	FP_MOV(c, y)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_26_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_26:
	FP_MOV(z, a)

	// fp_sub(X1, X1, ta); // th
	FP_MOV(b, x)
	FP_MOV(c, ta)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_27_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_27:
	FP_MOV(x, a)

	// fp_x3(ta, ta);      // ti
	FP_MOV(a, ta)
    // cal fp_x3
    mov.u32 %fun, fp_x3_;
    mov.u32 %ret, retg1p_addsub_28_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_28:
	FP_MOV(ta, a)

	// fp_x12(X1, X1);     // tn
	FP_MOV(a, x)
    // cal fp_x12
    mov.u32 %fun, fp_x12_;
    mov.u32 %ret, retg1p_addsub_29_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_29:
	FP_MOV(x, a)

// BLOCK 2
	// fp_add(td, Y2, Y1); // tl
	FP_MOV(b, v)
	FP_MOV(c, y)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_30_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_30:
	FP_MOV(td, a)

	// fp_sub(Y1, Y1, Y2); // tm
	FP_MOV(b, y)
	FP_MOV(c, v)
    // cal fp_sub
    mov.u32 %fun, fp_sub_;
    mov.u32 %ret, retg1p_addsub_31_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_31:
	FP_MOV(y, a)

// BLOCK 3
// Active (tag/var) = t7/tc, tc/Z1, ti/ta, tl/td, tm/Y1, tn/X1, T7/X2, Tc/Z2
// Available (var) = tb, Y2
	// fp_cpy(tb, X2); // T7
	FP_MOV(tb, u)

	// fp_mma(X2, tb, td, Z2, X1); // T7, -Tm=tl, Tc, tn
	FP_MOV(b, tb)
	FP_MOV(c, td)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_32_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_32:
	FP_MOV(b, w)
	FP_MOV(c, x)
	FP_MOV(x, a)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_33_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_33:
	FP_MOV(b, x)
	FP_MOV(x, c)
	FP_MOV(c, a)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_34_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_34:
	FP_MOV(u, a)

	// fp_neg(X2, X2); // X2
	FP_MOV(b, u)
    // cal fp_neg
    mov.u32 %fun, fp_neg_;
    mov.u32 %ret, retg1p_addsub_35_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_35:
	FP_MOV(u, a)

	// fp_neg(Z2, Z2); // -Tc
	FP_MOV(b, w)
    // cal fp_neg
    mov.u32 %fun, fp_neg_;
    mov.u32 %ret, retg1p_addsub_36_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_36:
	FP_MOV(w, a)

	// fp_mma(Z2, tb, ta, Z2, Y1); // T7, ti, -Tc, tm
	FP_MOV(b, tb)
	FP_MOV(c, ta)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_37_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_37:
	FP_MOV(b, w)
	FP_MOV(c, y)
	FP_MOV(y, a)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_38_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_38:
	FP_MOV(b, y)
	FP_MOV(y, c)
	FP_MOV(c, a)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_39_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_39:
	FP_MOV(w, a)

	// fp_mma(Y2, ta, X1, td, Y1); // ti, tn, tl, tm
	FP_MOV(b, ta)
	FP_MOV(c, x)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_40_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_40:
	FP_MOV(b, td)
	FP_MOV(c, y)
	FP_MOV(y, a)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_41_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_41:
	FP_MOV(b, y)
	FP_MOV(y, c)
	FP_MOV(c, a)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_42_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_42:
	FP_MOV(v, a)

	// fp_neg(X1, X1); // -tn
	FP_MOV(b, x)
    // cal fp_neg
    mov.u32 %fun, fp_neg_;
    mov.u32 %ret, retg1p_addsub_43_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_43:
	FP_MOV(x, a)

	// fp_mma(X1, tc, Y1, Z1, X1); // t7, tm, tc, -tn
	FP_MOV(b, tc)
	FP_MOV(c, y)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_44_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_44:
	FP_MOV(b, z)
	FP_MOV(c, x)
	FP_MOV(x, a)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_45_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_45:
	FP_MOV(b, x)
	FP_MOV(x, c)
	FP_MOV(c, a)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_46_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_46:
	FP_MOV(x, a)

	// fp_mma(Z1, tc, ta, Z1, td); // t7, ti, tc, tl
	FP_MOV(b, tc)
	FP_MOV(c, ta)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_47_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_47:
	FP_MOV(b, z)
	FP_MOV(c, td)
	FP_MOV(td, a)
    // cal fp_mul
    mov.u32 %fun, fp_mul_;
    mov.u32 %ret, retg1p_addsub_48_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_48:
	FP_MOV(b, td)
	FP_MOV(td, c)
	FP_MOV(c, a)
    // cal fp_add
    mov.u32 %fun, fp_add_;
    mov.u32 %ret, retg1p_addsub_49_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_49:
	FP_MOV(z, a)

// BLOCK 4
	// fp_cpy(Y1, Y2);
	FP_MOV(y, v)

// BLOCK 5
    // cal STORE2
    mov.u32 %fun, STORE2_;
    mov.u32 %ret, retg1p_addsub_50_;
    brx.idx.uni %fun, fp_fun;
retg1p_addsub_50:
	ret.uni;
