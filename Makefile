CXX=g++
NVCC=nvcc -rdc=true #--maxrregcount=64 #-Xlinker=--no-relax
NVOPTS=--compile
NVARCH= --gpu-architecture=compute_80 --gpu-code=sm_86 
COPTS= -O2

FP=fp fp_cpy fp_reduce6 fp_eq fp_neq fp_neg fp_x2 fp_x3 fp_x4 fp_x8 fp_x12 fp_add fp_sub fp_sqr fp_mul fp_inv fp_isone fp_iszero fp_nonzero fp_mma
FR=fr fr_cpy fr_reduce4 fr_eq fr_neq fr_neg fr_x2 fr_x3 fr_x4 fr_x8 fr_x12 fr_add fr_sub fr_sqr fr_mul fr_inv fr_isone fr_iszero fr_nonzero fr_roots fr_fft fr_addsub
G1=g1a g1p g1p_compare g1p_add g1p_dbl g1p_mul g1p_neg g1p_scale g1p_ispoint g1p_sub g1p_addsub g1p_fft
FK20=fk20 fk20_poly2h_fft fk20_poly2toeplitz_coefficients fk20_poly2toeplitz_coefficients_fft fk20_poly2hext_fft fk20_msm fk20_hext_fft2h_fft

FPTEST=fptest fptest_kat fptest_cmp fptest_mma fptest_inv fptest_add fptest_sub fptest_mul fptest_mulconst fptest_sqr fptest_distributive fptest_fibonacci
FRTEST=frtest frtest_kat frtest_cmp frtest_add frtest_mul frtest_sub frtest_addsub frtest_fibonacci frtest_mulconst frtest_distributive frtest_fft
G1TEST=g1test g1test_kat g1test_fibonacci g1test_dbl g1test_fft
FK20TEST=fk20test fk20test_poly fk20_testvector fk20test_fft fk20test_fft_rand
FK20TEST_TC=fk20test_poly2toeplitz_coefficients polynomial toeplitz_coefficients
FK20TEST_TCFFT=fk20test_poly2toeplitz_coefficients_fft polynomial toeplitz_coefficients_fft
FFTTEST=fftTest parseFFTTest
FK20_512TEST=fk20_512test xext_fft polynomial toeplitz_coefficients toeplitz_coefficients_fft hext_fft h h_fft
FK20BENCHMARK=fk20benchmark fk20_testvector

FP_OBJS=$(FP:%=%.o)
FR_OBJS=$(FR:%=%.o)
G1_OBJS=$(G1:%=%.o)
FK20_OBJS=$(FK20:%=%.o)
FK20_OBJS=$(FK20:%=%.o)

FP_CUBIN=$(FP:%=%.cubin)
FR_CUBIN=$(FR:%=%.cubin)
G1_CUBIN=$(G1:%=%.cubin)
FK20_CUBIN=$(FK20:%=%.cubin)

FPTEST_OBJS=$(FPTEST:%=%.o)
FRTEST_OBJS=$(FRTEST:%=%.o)
G1TEST_OBJS=$(G1TEST:%=%.o)
FK20TEST_OBJS=$(FK20TEST:%=%.o)
FK20TEST_TC_OBJS=$(FK20TEST_TC:%=%.o)
FK20TEST_TCFFT_OBJS=$(FK20TEST_TCFFT:%=%.o)
FFTTEST_OBJS=$(FFTTEST:%=%.o)
FK20_512TEST_OBJS=$(FK20_512TEST:%=%.o)
FK20BENCHMARK_OBJS=$(FK20BENCHMARK:%=%.o)

OBJS=$(FP_OBJS) $(FR_OBJS) $(G1_OBJS) $(FK20_OBJS)
CUBIN=$(FP_CUBIN) $(FR_CUBIN) $(G1_CUBIN) $(FK20_CUBIN)
TEST_OBJS=$(FPTEST_OBJS) $(FRTEST_OBJS) $(G1TEST_OBJS) $(FK20TEST_OBJS) $(FK20_512TEST_OBJS)

all: fptest frtest g1test fk20test ffttest fk20_512test fk20test_poly2toeplitz_coefficients fk20test_poly2toeplitz_coefficients_fft

#add some debug flags. 
debug: 
	$(eval NVCC += -g -G --maxrregcount=128 -DDEBUG)

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
	-rm -f xext_fft.cu polynomial.cu toeplitz_coefficients.cu toeplitz_coefficients_fft.cu hext_fft.cu h.cu h_fft.cu

shallowclean:
	@(echo "Removing only objects that are fast to compile!")
	-rm -f $(OBJS) $(CUBIN)

clobber: clean
	-rm -f fptest frtest g1test fk20test fk20_512test fk20test_poly2toeplitz_coefficients fk20test_poly2toeplitz_coefficients_fft

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

fk20test_poly2toeplitz_coefficients: $(FK20TEST_TC_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ -G # --resource-usage

fk20test_poly2toeplitz_coefficients_fft: $(FK20TEST_TCFFT_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ -G # --resource-usage

ffttest: $(FFTTEST_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

#use this rule to generate the large objects without debug symbols
fk20_512test_objs: $(FK20_512TEST_OBJS)

#use this rule to remake objects
fk20_objs: $(OBJS)

fk20_512test: $(FK20_512TEST_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

fk20benchmark: $(FK20BENCHMARK_OBJS) $(OBJS)
	$(NVCC) $(NVARCH) -o $@ $^ # --resource-usage

fp%.cubin: fp%.cu fp.cuh
	$(NVCC) $(NVOPTS) --gpu-architecture=sm_86 -o $@ -c $< -cubin

fr%.cubin: fr%.cu fr.cuh
	$(NVCC) $(NVOPTS) --gpu-architecture=sm_86 -o $@ -c $< -cubin

g1%.cubin: g1%.cu g1.cuh fp.cuh fr.cuh
	$(NVCC) $(NVOPTS) --gpu-architecture=sm_86 -o $@ -c $< -cubin

##############################
#
# Test vector generation
#
##############################

512:=$(shell ./512.sh)

define ROW_template =
 test/fk20test-fib-1-$(1).cu: FK20Py/fk20_multi_cuda.py FK20Py/fk20_single_cuda.py
	-mkdir -p test
	$$< 1 $(1) > $$@
 ALL_ROWS += test/fk20test-fib-1-$(1).cu
endef

$(foreach i,$512,$(eval $(call ROW_template,$i)))

testvector: $(ALL_ROWS)

xext_fft.cu: test/fk20test-fib-1-0.cu
	(echo \#include \"g1.cuh\"; echo; grep -A 40993 -B1 xext_fft $< ) > $@

polynomial.cu: $(ALL_ROWS)
	./polynomial.sh > $@

toeplitz_coefficients.cu: $(ALL_ROWS)
	./toeplitz_coefficients.sh > $@

toeplitz_coefficients_fft.cu: $(ALL_ROWS)
	./toeplitz_coefficients_fft.sh > $@

hext_fft.cu: $(ALL_ROWS)
	./hext_fft.sh > $@

h.cu: $(ALL_ROWS)
	./h.sh > $@

h_fft.cu: $(ALL_ROWS)
	./h_fft.sh > $@

