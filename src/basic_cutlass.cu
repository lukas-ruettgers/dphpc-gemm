// gemm_blocked_example.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>
#include <vector>

// --- CUTLASS HEADERS ---
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef BENCHMARK
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define BENCHMARK 1
#endif

#if BENCHMARK
float WARMUP=3;
float ITER=50;
#endif

// simple CUDA error-check macro
inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// Matrix sizes (runtime)
int M = 1024;
int K = 1024;
int N = 1024;

// Derived block counts (runtime)
int BM, BK, BN;

// --- CUTLASS DEFINITION ---
using CutlassGemm = cutlass::gemm::device::Gemm<
    float,                           // ElementA
    cutlass::layout::RowMajor,       // LayoutA
    float,                           // ElementB
    cutlass::layout::RowMajor,       // LayoutB
    float,                           // ElementC
    cutlass::layout::RowMajor,       // LayoutC
    float,                           // ElementAccumulator
    cutlass::arch::OpClassSimt,      // Tag for CUDA Core execution
    cutlass::arch::Sm80              // Adjust this to your GPU (Sm75, Sm80, etc)
>;

// CPU naive gemm for verification: C = A * B
void cpu_gemm_naive(const float* A, const float* B, float* C) {
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

int main(int argc, char** argv) {
    if (argc == 4) {
        M    = std::atoi(argv[1]);
        N    = std::atoi(argv[2]);
        K    = std::atoi(argv[3]);
    } else {
        std::cout << "Usage: ./gemm M N K\n";
        std::cout << "Using default values.\n";
    }

    std::cout << "Non-Blocked GEMM end-to-end example (CUTLASS Backend)\n";
    std::cout << "M=" << M << " K=" << K << " N=" << N << "\n";

    // Allocate host row-major matrices
    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    float* hA = (float*)malloc(sizeA * sizeof(float));
    float* hB = (float*)malloc(sizeB * sizeof(float));
    float* hC = (float*)malloc(sizeC * sizeof(float));
    float* hC_ref = (float*)malloc(sizeC * sizeof(float));

    // Fill random data
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) hA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hB[i] = dist(rng);

    // Device allocations
    float *dAblk = nullptr, *dBblk = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dAblk, sizeA * sizeof(float)));
    cudaCheck(cudaMalloc(&dBblk, sizeB * sizeof(float)));
    cudaCheck(cudaMalloc(&dC, sizeC * sizeof(float)));

    // Copy blocked tensors to device
    cudaCheck(cudaMemcpy(dAblk, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dBblk, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dC, 0, sizeC * sizeof(float)));

    // --- CUTLASS SETUP ---
    CutlassGemm gemm_op;
    typename CutlassGemm::Arguments args(
        {M, N, K},          // Problem dimensions
        {dAblk, K},         // TensorRef A (ptr, leading dimension)
        {dBblk, N},         // TensorRef B
        {dC, N},            // TensorRef C
        {dC, N},            // TensorRef D (output)
        {1.0f, 0.0f}        // Alpha, Beta
    );

    // Check if initialization succeeded
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize CUTLASS gemm: " 
                  << cutlass::cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    // Initial Launch
    status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS Error: %s\n", cutlass::cutlassGetStatusString(status));
        return 1;
    }
    cudaCheck(cudaDeviceSynchronize());

    // Copy result back
    cudaCheck(cudaMemcpy(hC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    #if CPU_DEBUG
    // CPU reference
    cpu_gemm_naive(hA, hB, hC_ref);

    // Verify
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double d = fabs((double)hC_ref[i] - (double)hC[i]);
        sum_abs_diff += d;
        if (d > max_abs_diff) max_abs_diff = d;
    }

    std::cout << "Max absolute difference: " << max_abs_diff << "\n";
    std::cout << "Sum absolute difference: " << sum_abs_diff << "\n";

    const double eps = 1e-3; // tolerance (floating rounding)
    if (max_abs_diff < eps) {
        std::cout << "PASS: GPU result matches CPU reference within eps=" << eps << "\n";
    } else {
        std::cout << "FAIL: difference exceeds eps=" << eps << "\n";
    }
    #endif

    #if BENCHMARK
    // --------------------------
    // Timing (CUDA events)
    // --------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -----------------------------
    // Warm-up
    // -----------------------------
    for (int i = 0; i < WARMUP; i++) {
        gemm_op(args);
    }
    cudaDeviceSynchronize();

    // -----------------------------
    // Benchmark
    // -----------------------------
    std::vector<float> times_ms;
    times_ms.reserve(ITER);

    float total_ms = 0.f, min_ms = 1e9f, max_ms = 0.f;

    for (int i = 0; i < ITER; i++) {
        cudaEventRecord(start);

        gemm_op(args);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);

        times_ms.push_back(ms);

        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    float mean_ms = total_ms / ITER;

    // ---- Compute sample standard deviation ----
    float var_ms = 0.f;
    for (float t : times_ms) {
        float diff = t - mean_ms;
        var_ms += diff * diff;
    }
    var_ms /= (ITER - 1);              
    float std_ms = std::sqrt(var_ms);

    float stderr_ms = std_ms / std::sqrt((float)ITER);
    float ci95_ms = 1.96f * stderr_ms;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (mean_ms / 1000.0) / 1e9;
    double gflops_ci95 = gflops * (ci95_ms / mean_ms);

    std::cout << "---- Benchmark ----\n";
    std::cout << "Mean time: " << mean_ms << " ms\n";
    std::cout << "Min time:  " << min_ms << " ms\n";
    std::cout << "Max time:  " << max_ms << " ms\n";
    std::cout << "Stddev:    " << std_ms << " ms\n";

    std::cout << "Achieved:  " << gflops
            << " ± " << gflops_ci95
            << " GFLOP/s (95% CI)\n";
    #endif

    // Cleanup
    free(hA); free(hB); free(hC); free(hC_ref);
    cudaFree(dAblk); cudaFree(dBblk); cudaFree(dC);

    return 0;
}