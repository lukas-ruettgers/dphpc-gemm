CUTLASS_DIR=/home/lruettgers/dphpc/dphpc-gemm/external/cutlass
NVCC=nvcc
XFLAGS=-Xptxas=--verbose -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -Xcompiler -ftemplate-backtrace-limit=0 -Xcompiler -fno-elide-type
XFLAGS_EXTENDED=-Xcompiler=-g -Xcompiler=-Og -Xcompiler=-fno-omit-frame-pointer -Xcompiler=-fno-optimize-sibling-calls
DEFINES=-DNDEBUG 
ifeq ($(DEBUGDUMP),1)
	DEFINES += -DDEBUGDUMP
endif
ifeq ($(DPHPC),1)
	DEFINES += -DDPHPC
endif
INCLUDES=-I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/tools/util/include
ARCHSM80=-gencode arch=compute_80,code=\"sm_80,compute_80\"
ARCHSM90=-gencode arch=compute_90,code=\"sm_90,compute_90\"
ARCHSM120=-gencode arch=compute_120a,code=\"sm_120a,compute_120a\"
NVCCFLAGS=-std=c++17 -O3 ${XFLAGS} ${XFLAGS_EXTENDED} ${INCLUDES} \
	--expt-relaxed-constexpr

LDFLAGS=
LDLIBS=-lcuda

main: gemm_streamk_ref.cu
	${NVCC} ${NVCCFLAGS} -lineinfo ${ARCHSM80} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS}
tb160: gemm_streamk_ref.cu
	${NVCC} ${NVCCFLAGS} -lineinfo ${ARCHSM80} ${DEFINES} -DTB160 ${LDFLAGS} -o $@ $< ${LDLIBS}
tb192: gemm_streamk_ref.cu
	${NVCC} ${NVCCFLAGS} -lineinfo ${ARCHSM80} ${DEFINES} -DTB192 ${LDFLAGS} -o $@ $< ${LDLIBS}

props: device_props.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM120} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS}

sm120: gemm_sm120.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM120} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS}
sm90_ref: gemm_sm90_ref.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM90} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS}
access_width: access_width_exceeded.cu
	${NVCC} ${NVCCFLAGS} -lineinfo ${ARCHSM80} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS}

DEFINESMNK=-DDPC_M=32 -DDPC_N=32 -DDPC_K=16
layouts: layouts.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM80} ${DEFINES} ${DEFINESMNK} ${LDFLAGS} -o $@ $< ${LDLIBS}
movement: data-movement.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM80} ${DEFINES} ${DEFINESMNK} ${LDFLAGS} -o $@ $< ${LDLIBS}
swizzle: swizzle.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM80} ${DEFINES} ${DEFINESMNK} ${LDFLAGS} -o $@ $< ${LDLIBS}

testing: testing.cu
	${NVCC} ${NVCCFLAGS} ${ARCHSM90} ${DEFINES} ${LDFLAGS} -o $@ $< ${LDLIBS} -lcublas
