#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

//============================
// Active Kernels
//============================

#define KERNEL_V00_BASIC

//============================
// Structs
//============================

struct KernelArgs {
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

//============================
// Kernels
//============================

#ifdef KERNEL_V00_BASIC
__global__ void kernel_v00_basic(KernelArgs args);
#endif

//============================
// Functions
//============================

// simple CUDA error-check macro
static inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}