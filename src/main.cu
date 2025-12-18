#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>
#include "kernels.hpp"

#if GEMM_BENCHMARK
int WARMUP=3;
int ITER=50;
#endif

// CPU naive gemm for verification: C = A * B
static void cpu_gemm_naive(const float* A, const float* B, float* C, size_t M, size_t N, size_t K);
static KernelArgs parse_args(int argc, char** argv);


int main(int argc, char** argv) {
    KernelArgs kargs = parse_args(argc, argv);
    UNPACK_KERNEL_ARGS(kargs);

    printf("\nStarting GEMM framework...\n");
    printf("(M, N, K) = (%lu, %lu, %lu)\n", M, N, K);
    printf("(TB_M, TB_N, TB_K) = (%lu, %lu, %lu)\n", TB_M, TB_N, TB_K);
    printf("(blocks_M, blocks_N, blocks_K) = (%lu, %lu, %lu)\n", blocks_M, blocks_N, blocks_K);

    // Allocate host row-major matrices
    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    float* hostA = (float*) malloc(sizeA * sizeof(float));
    float* hostB = (float*) malloc(sizeB * sizeof(float));
    float* hostC = (float*) malloc(sizeC * sizeof(float));
    float* hostC_ref = (float*) malloc(sizeC * sizeof(float));

    // Fill random data
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) hostA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hostB[i] = dist(rng);

    // Device allocations
    float *devA = nullptr, *devB = nullptr, *devC = nullptr;
    cudaCheck(cudaMalloc(&devA, sizeA * sizeof(float)));
    cudaCheck(cudaMalloc(&devB, sizeB * sizeof(float)));
    cudaCheck(cudaMalloc(&devC, sizeC * sizeof(float)));

    // Set device pointers in kernel args.
    kargs.A = devA;
    kargs.B = devB;
    kargs.C = devC;

    // Copy blocked tensors to device
    cudaCheck(cudaMemcpy(devA, hostA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(devB, hostB, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(devC, 0, sizeC * sizeof(float)));

    // Launch kernel
    // dim3 grid(blocks_M, blocks_N); // 2D grid of blocks_M x blocks_N blocks
    // dim3 block(TB_M, TB_N); // 2D block of TB_M x TB_N threads
    // size_t smem_bytes = (TB_M * TB_K + TB_K * TB_N) * sizeof(float);
    // gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K, TB_M, TB_N, TB_K, BM, BN, BK);

    printf("\nLaunching kernel...\n");
    launch_kernel(kargs);

    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // Copy result back
    cudaCheck(cudaMemcpy(hostC, devC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    #if GEMM_VERIFY
    // CPU reference
    cpu_gemm_naive(hostA, hostB, hostC_ref, M, N, K);

    // Verify
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double d = fabs((double) hostC_ref[i] - (double) hostC[i]);
        sum_abs_diff += d;
        if (d > max_abs_diff) max_abs_diff = d;
    }

    printf("\nMax absolute difference: %f\n", max_abs_diff);
    printf("Sum absolute difference: %f\n", sum_abs_diff);

    const double eps = 1e-3; // tolerance (floating rounding)
    if (max_abs_diff < eps) {
        printf("\x1b[32mPASS\x1b[0m: GPU result matches CPU reference within eps = %f\n", eps);
    } else {
        printf("\x1b[31mFAIL\x1b[0m: difference exceeds eps = %f\n", eps);
        exit(1);
    }

    #endif


    #if GEMM_BENCHMARK
    // --------------------------
    // Timing (CUDA events)
    // --------------------------
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -----------------------------
    // Warm-up (5 iterations)
    // -----------------------------
    for (int i = 0; i < WARMUP; i++) {
        gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K, TB_M, TB_N, TB_K, BM, BN, BK);
    }
    cudaDeviceSynchronize();

    // -----------------------------
    // Benchmark (10 runs)
    // -----------------------------
    float total_ms = 0.f, min_ms = 1e9, max_ms = 0.f;

    std::cout << "-----BEGIN\n";
    for (int i = 0; i < ITER; i++) {
        cudaEventRecord(start);

        gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K, TB_M, TB_N, TB_K, BM, BN, BK);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);

        std::cout << ms << "\n";
    }
    std::cout << "-----END\n";

    float avg_ms = total_ms / ITER;

    std::cout << "---- Benchmark ----\n";
    std::cout << "Avg time: " << avg_ms << " ms\n";
    std::cout << "Min time: " << min_ms << " ms\n";
    std::cout << "Max time: " << max_ms << " ms\n";

    double gflops = (2.0 * M * N * K) / (avg_ms/1000.0) / 1e9;
    std::cout << "Achieved: " << gflops << " GFLOP/s\n";

    #endif

    // Cleanup
    free(hostA); free(hostB); free(hostC); free(hostC_ref);
    cudaFree(devA); cudaFree(devB); cudaFree(devC);

    return 0;
}


// CPU naive gemm for verification: C = A * B
static void cpu_gemm_naive(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}


static KernelArgs parse_args(int argc, char** argv) {
    if (argc != 8) {
        printf("Usage: ./gemm <kernel> M N K TB_M TB_N TB_K\n");
        printf("Exiting....\n");
        exit(1);
    }

    // Parse kernel version.
    KernelVersion kernel = KernelVersion_from_string(argv[1]);
    if (kernel == KernelVersion::Invalid) {
        printf("\x1b[31mERROR\x1b[0m: Invalid kernel version: %s\n", argv[1]);
        exit(1);
    }

    size_t M = (size_t) atoi(argv[2]);
    size_t N = (size_t) atoi(argv[3]);
    size_t K = (size_t) atoi(argv[4]);
    size_t TB_M = (size_t) atoi(argv[5]);
    size_t TB_N = (size_t) atoi(argv[6]);
    size_t TB_K = (size_t) atoi(argv[7]);
    return KernelArgs {
        .kernel = kernel,
        .M = M,
        .N = N,
        .K = K,
        .TB_M = TB_M,
        .TB_N = TB_N,
        .TB_K = TB_K,
        .blocks_M = M / TB_M,
        .blocks_N = N / TB_N,
        .blocks_K = K / TB_K,
    };
}