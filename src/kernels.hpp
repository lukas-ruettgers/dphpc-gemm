#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

//============================
// MACRO PARAMETERS
//============================

#define GEMM_VERIFY 1
#define GEMM_BENCHMARK 1

#define WARMUP_RUNS 2
#define BENCHMARK_RUNS 5

//================
// Kernels
//================

// Helper macros to define and declare kernel instances.
#define KERNEL_INSTANCE(kernel_class) kernel_instance_##kernel_class
#define DEFINE_KERNEL_INSTANCE(kernel_class) kernel_class KERNEL_INSTANCE(kernel_class)
#define DECLARE_KERNEL(kernel_class) class kernel_class; extern DEFINE_KERNEL_INSTANCE(kernel_class);


//======================================================================================
// DECLARE NEW KERNELS HERE
//======================================================================================

DECLARE_KERNEL(Kernel_V00_Basic);
DECLARE_KERNEL(Kernel_V01_Coalesced);

//======================================================================================


struct KernelContext;

class Kernel {
    public:
    virtual void launch(KernelContext ctx) = 0;
};

struct KernelContext {
    Kernel *kernel;

    const float *A;     // M x K
    const float *B;     // K x N
    float *C;           // M x N

    size_t M;
    size_t N;
    size_t K;

    size_t TB_M;
    size_t TB_N;
    size_t TB_K;

    size_t blocks_M; // M / TB_M
    size_t blocks_N; // N / TB_N
    size_t blocks_K; // K / TB_K
};

//==== Kernel Version Parsing ====

static inline Kernel *Kernel_from_string(const char *str) {
    struct KernelMap {
        const char* name;
        Kernel *kernel;
    };

    #define MAP_ENTRY(str, kernel_class) {str, (Kernel *) &KERNEL_INSTANCE(kernel_class)}

    const KernelMap kernel_map[] = {

//======================================================================================
// ADD NEW KERNEL MAP ENTRIES HERE
//======================================================================================

        MAP_ENTRY("v00_basic", Kernel_V00_Basic),
        MAP_ENTRY("v01_coalesced", Kernel_V01_Coalesced),

//======================================================================================

    };

    #undef MAP_ENTRY

    for (const auto& entry : kernel_map) {
        if (strcmp(str, entry.name) == 0) {
            return entry.kernel;
        }
    }

    return nullptr;
}

//============================
// Helpers
//============================

#define UNPACK_KERNEL_ARGS(args) \
    const float* A = args.A; \
    const float* B = args.B; \
    float* C = args.C; \
    \
    size_t M = args.M; \
    size_t N = args.N; \
    size_t K = args.K; \
    \
    size_t TB_M = args.TB_M; \
    size_t TB_N = args.TB_N; \
    size_t TB_K = args.TB_K; \
    \
    size_t blocks_M = args.blocks_M; \
    size_t blocks_N = args.blocks_N; \
    size_t blocks_K = args.blocks_K;


// simple CUDA error-check macro
static inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}