CXX=g++
NVCC=nvcc -rdc=true
NVOPTS=--compile
NVARCH= --gpu-architecture=compute_80 --gpu-code=sm_86
COPTS=-O2

FP_OBJS=fp.o fp_cpy.o fp_reduce6.o fp_eq.o fp_neq.o fp_neg.o fp_x2.o fp_x3.o fp_x4.o fp_x8.o fp_x12.o fp_add.o fp_sub.o fp_sqr.o fp_mul.o fp_inv.o fp_isone.o fp_iszero.o fp_nonzero.o fp_mma.o
FR_OBJS=fr.o fr_cpy.o fr_reduce4.o fr_eq.o fr_neq.o fr_neg.o fr_x2.o fr_x3.o fr_x4.o fr_x8.o fr_x12.o fr_add.o fr_sub.o fr_sqr.o fr_mul.o fr_inv.o fr_isone.o fr_iszero.o
G1_OBJS=g1a.o g1p.o g1p_compare.o g1p_add.o g1p_dbl.o g1p_mul.o g1p_neg.o g1p_scale.o g1p_ispoint.o g1p_sub.o g1p_addsub.o
FK20_OBJS=fk20_fft.o
OBJS=$(FP_OBJS) $(FR_OBJS) $(G1_OBJS) $(FK20_OBJS)

FPTEST_OBJS=fptest.o fptest_kat.o fptest_cmp.o fptest_mma.o
FRTEST_OBJS=frtest.o frtest_kat.o frtest_cmp.o
G1TEST_OBJS=g1test.o g1test_kat.o g1test_fibonacci.o
FK20TEST_OBJS=fk20test.o fk20test_kat.o
TEST_OBJS=$(FPTEST_OBJS) $(FRTEST_OBJS) $(G1TEST_OBJS) $(FK20TEST_OBJS)

FP_CUBIN=fp_cpy.cubin fp_reduce6.cubin fp_eq.cubin fp_neq.cubin fp_neg.cubin fp_x2.cubin fp_x3.cubin fp_x4.cubin fp_x8.cubin fp_x12.cubin fp_add.cubin fp_sub.cubin fp_sqr.cubin fp_mul.cubin fp_inv.cubin fp_isone.cubin fp_iszero.cubin
FR_CUBIN=fr_cpy.cubin fr_reduce4.cubin fr_eq.cubin fr_neq.cubin fr_neg.cubin fr_x2.cubin fr_x3.cubin fr_x4.cubin fr_x8.cubin fr_x12.cubin fr_add.cubin fr_sub.cubin fr_sqr.cubin fr_mul.cubin fr_inv.cubin fr_isone.cubin fr_iszero.cubin
G1_CUBIN=g1a.cubin g1p.cubin g1p_compare.cubin g1p_add.cubin g1p_dbl.cubin g1p_mul.cubin g1p_neg.cubin g1p_scale.cubin g1p_ispoint.cubin g1p_sub.cubin
FK20_CUBIN=fk20_fft.cubin
CUBIN=$(FP_CUBIN) $(FR_CUBIN) $(G1_CUBIN) $(FK20_CUBIN)

all: fptest frtest g1test fk20test # $(OBJS) $(TEST_OBJS)

run: fp-run fr-run g1-run fk20-run

fp-run: fptest
	./fptest

fr-run: frtest
	./frtest

g1-run: g1test
	./g1test

fk20-run: fk20test
	./fk20test

cubin: $(CUBIN)

clean:
	-rm -f $(OBJS) $(TEST_OBJS) $(CUBIN)

clobber: clean
	-rm -f fptest frtest g1test fk20test

%.o: %.cu
	$(NVCC) $(NVOPTS) $(NVARCH) -o $@ -c $<

%: %.o
	$(NVCC) $(NVARCH) -o $@ $^ --resource-usage

fp.o: fp.cu fp.cuh
	$(NVCC) $(NVOPTS) $(NVARCH) -o $@ -c $<

fr.o: fr.cu fr.cuh
	$(NVCC) $(NVOPTS) $(NVARCH) -o $@ -c $<

fp%.o: fp%.cu fp.cuh
	$(NVCC) $(NVOPTS) $(NVARCH) -o $@ -c $<

fr%.o: fr%.cu fr.cuh
	$(NVCC) $(NVOPTS) $(NVARCH) -o $@ -c $<

g1%.o: g1%.cu g1.cuh fp.cuh fr.cuh
	$(NVCC) $(NVOPTS) $(NVARCH) -o $@ -c $<

fptest_%.o: fptest_%.cu fptest.cuh
	$(NVCC) $(NVOPTS) $(NVARCH) -o $@ -c $<

frtest_%.o: frtest_%.cu frtest.cuh
	$(NVCC) $(NVOPTS) $(NVARCH) -o $@ -c $<

fptest.o: fptest.cu fp.cuh
	$(NVCC) $(COPTS) -o $@ -c $<

frtest.o: frtest.cu fr.cuh
	$(NVCC) $(COPTS) -o $@ -c $<

g1test.o: g1test.cu g1.cuh fp.cuh fr.cuh
	$(NVCC) $(COPTS) -o $@ -c $<

fk20test.o: fk20test.cu fk20.cuh g1.cuh fp.cuh fr.cuh
	$(NVCC) $(COPTS) -o $@ -c $<

fptest: $(FPTEST_OBJS) $(FP_OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

frtest: $(FRTEST_OBJS) $(FR_OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

g1test: $(G1TEST_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

fk20test: $(FK20TEST_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

fp%.cubin: fp%.cu fp.cuh
	$(NVCC) $(NVOPTS) --gpu-architecture=sm_86 -o $@ -c $< -cubin

fr%.cubin: fr%.cu fr.cuh
	$(NVCC) $(NVOPTS) --gpu-architecture=sm_86 -o $@ -c $< -cubin

g1%.cubin: g1%.cu g1.cuh fp.cuh fr.cuh
	$(NVCC) $(NVOPTS) --gpu-architecture=sm_86 -o $@ -c $< -cubin
