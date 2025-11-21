CUTLASS_DIR=/home/lruettgers/dphpc/dphpc-gemm/external/cutlass/
NVCC=nvcc
XFLAGS=-Xptxas=--verbose -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
DEFINES=-DNDEBUG
INCLUDES=-I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/tools/util/include
ARCHSM80=-gencode arch=compute_80,code=\"sm_80,compute_80\"
ARCHSM90=-gencode arch=compute_90,code=\"sm_90,compute_90\"
ARCHSM120=-gencode arch=compute_120a,code=\"sm_120a,compute_120a\"
NVCCFLAGS=-std=c++17 -O3 ${XFLAGS} ${INCLUDES} \
	--expt-relaxed-constexpr

LDFLAGS=
LDLIBS=-lcuda

props: device_props.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM120} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS}

sm120: gemm_sm120.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM120} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS}
sm90_ref: gemm_sm90_ref.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM90} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS}
sm80_streamk: gemm_streamk_ref.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM80} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS}

testing: testing.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM90} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS} -lcublas