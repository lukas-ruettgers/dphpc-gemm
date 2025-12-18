#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

//============================
// Macros
//============================

#define GEMM_VERIFY 1
#define GEMM_BENCHMARK 1

#define WARMUP_RUNS 2
#define BENCHMARK_RUNS 5

//============================
// Kernels
//============================

//==== Kernel Versions ====

// ADD: new kernel versions
enum class KernelVersion {
    Invalid = 0,
    V00_Basic,
};


struct KernelArgs {
    KernelVersion kernel;

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


//==== Kernel Declarations ====

// ADD: new kernel declarations
__global__ void kernel_v00_basic(KernelArgs);

//==== Kernel Launches ====

static inline void launch_kernel(KernelArgs kargs) {
    #define KERNEL_CASE(version, func) \
        case KernelVersion::version: func<<<grid, block>>>(kargs); break;

    dim3 grid(kargs.blocks_M, kargs.blocks_N); // 2D grid of blocks_M x blocks_N blocks
    dim3 block(kargs.TB_M, kargs.TB_N); // 2D block of TB_M x TB_N threads

    switch (kargs.kernel) {
        // ADD: new kernel launches
        KERNEL_CASE(V00_Basic, kernel_v00_basic);
        
        default:
            fprintf(stderr, "ERROR: Unsupported kernel version.\n");
            exit(1);
    }
}

//==== Kernel Version Parsing ====

static inline KernelVersion KernelVersion_from_string(const char *str) {
    struct KernelVersionMap {
        const char* name;
        KernelVersion version;
    };

    // ADD: new kernel name mappings
    const KernelVersionMap version_map[] = {
        {"v00_basic", KernelVersion::V00_Basic},
    };

    for (const auto& entry : version_map) {
        if (strcmp(str, entry.name) == 0) {
            return entry.version;
        }
    }
    return KernelVersion::Invalid;
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