CXX=g++
NVCC=nvcc -rdc=true #-g -G -O0
NVOPTS=--compile
NVARCH= --gpu-architecture=compute_80 --gpu-code=sm_86
COPTS= -O2

FP=fp fp_cpy fp_reduce6 fp_eq fp_neq fp_neg fp_x2 fp_x3 fp_x4 fp_x8 fp_x12 fp_add fp_sub fp_sqr fp_mul fp_inv fp_isone fp_iszero fp_nonzero fp_mma
FR=fr fr_cpy fr_reduce4 fr_eq fr_neq fr_neg fr_x2 fr_x3 fr_x4 fr_x8 fr_x12 fr_add fr_sub fr_sqr fr_mul fr_inv fr_isone fr_iszero fr_nonzero fr_roots fr_fft fr_addsub
G1=g1a g1p g1p_compare g1p_add g1p_dbl g1p_mul g1p_neg g1p_scale g1p_ispoint g1p_sub g1p_addsub g1p_fft
FK20=fk20

FPTEST=fptest fptest_kat fptest_cmp fptest_mma fptest_inv fptest_add fptest_sub fptest_mul fptest_mulconst fptest_sqr
FRTEST=frtest frtest_kat frtest_cmp frtest_add frtest_mul frtest_sub frtest_addsub frtest_fibonacci frtest_mulconst frtest_distributive frtest_fft
G1TEST=g1test g1test_kat g1test_fibonacci
FK20TEST=fk20test fk20test_poly fk20_testvector fk20test_fft fk20test_fft_rand
FFTTEST=fftTest parseFFTTest

FP_OBJS=$(FP:%=%.o)
FR_OBJS=$(FR:%=%.o)
G1_OBJS=$(G1:%=%.o)
FK20_OBJS=$(FK20:%=%.o)

FP_CUBIN=$(FP:%=%.cubin)
FR_CUBIN=$(FR:%=%.cubin)
G1_CUBIN=$(G1:%=%.cubin)
FK20_CUBIN=$(FK20:%=%.cubin)

FPTEST_OBJS=$(FPTEST:%=%.o)
FRTEST_OBJS=$(FRTEST:%=%.o)
G1TEST_OBJS=$(G1TEST:%=%.o)
FK20TEST_OBJS=$(FK20TEST:%=%.o)
FFTTEST_OBJS=$(FFTTEST:%=%.o)

OBJS=$(FP_OBJS) $(FR_OBJS) $(G1_OBJS) $(FK20_OBJS)
CUBIN=$(FP_CUBIN) $(FR_CUBIN) $(G1_CUBIN) $(FK20_CUBIN)
TEST_OBJS=$(FPTEST_OBJS) $(FRTEST_OBJS) $(G1TEST_OBJS) $(FK20TEST_OBJS)

all: fptest frtest g1test fk20test ffttest# $(OBJS) $(TEST_OBJS)

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

fp_add.o: fp_add.cu fp_add.cuh

fp_addsub.o: fp_addsub.cu fp_add.cuh

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
	$(NVCC) $(NVOPTS) -o $@ -c $<

frtest.o: frtest.cu fr.cuh
	$(NVCC) $(NVOPTS) -o $@ -c $<

g1test.o: g1test.cu g1.cuh fp.cuh fr.cuh
	$(NVCC) $(NVOPTS) -o $@ -c $<

fk20test.o: fk20test.cu fk20.cuh g1.cuh fp.cuh fr.cuh
	$(NVCC) $(NVOPTS) -o $@ -c $<

parseFFTTest.o: parseFFTTest.c
	gcc -g3 -ggdb $(COPTS) -o $@ -c $<

ffttest.o: fftTest.cu fk20.cuh g1.cuh fp.cuh fr.cuh parseFFTTest.c
	$(NVCC) $(COPTS) -o $@ -c $<

fptest: $(FPTEST_OBJS) $(FP_OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

frtest: $(FRTEST_OBJS) $(FR_OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

g1test: $(G1TEST_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

fk20test: $(FK20TEST_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

ffttest: $(FFTTEST_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

fp%.cubin: fp%.cu fp.cuh
	$(NVCC) $(NVOPTS) --gpu-architecture=sm_86 -o $@ -c $< -cubin

fr%.cubin: fr%.cu fr.cuh
	$(NVCC) $(NVOPTS) --gpu-architecture=sm_86 -o $@ -c $< -cubin

g1%.cubin: g1%.cu g1.cuh fp.cuh fr.cuh
	$(NVCC) $(NVOPTS) --gpu-architecture=sm_86 -o $@ -c $< -cubin
