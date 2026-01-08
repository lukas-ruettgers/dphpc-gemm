#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cutlass/numeric_types.h>
#include <cstring>

#include "gemm_cpasync_bk32.h"
#include "gemm_cpasync.h"
#include "gemm_pipelined.h"
#include "gemm_vector.h"
#include "gemm_scalar.h"

using cutlass::half_t;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

void cpu_gemm_reference(const half_t* A, const half_t* B, float* C,
                        int M, int N, int K, int lda, int ldb, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<float>(A[m * lda + k]) * static_cast<float>(B[n * ldb + k]);
            }
            C[m * ldc + n] = sum;
        }
    }
}

bool check_results(const float* C_gpu, const float* C_cpu, int M, int N, int ldc) {
    int errors = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int idx = m * ldc + n;
            float diff = std::abs(C_gpu[idx] - C_cpu[idx]);
            float rel_diff = diff / (std::abs(C_cpu[idx]) + 1e-8f);
            if (diff > 1e-2f && rel_diff > 1e-3f) {
                if (errors < 10) {
                    printf("Error at [%d,%d]: GPU=%.4f CPU=%.4f\n", m, n, C_gpu[idx], C_cpu[idx]);
                }
                errors++;
            }
        }
    }
    if (errors > 0) printf("Total errors: %d / %d\n", errors, M * N);
    return errors == 0;
}

void initialize_matrices(std::vector<half_t>& A, std::vector<half_t>& B) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& v : A) v = half_t(dis(gen));
    for (auto& v : B) v = half_t(dis(gen));
}

template<typename Func>
void benchmark_kernel(Func kernel_func, half_t* d_A, half_t* d_B, float* d_C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    const int WARMUP = 20, ITERS = 100;
    
    for (int i = 0; i < WARMUP; ++i) {
        kernel_func(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<float> timings;
    timings.reserve(ITERS);
    
    for (int i = 0; i < ITERS; ++i) {
        cudaEventRecord(start);
        kernel_func(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        timings.push_back(ms);
    }
    
    float sum = 0.0f;
    for (float t : timings) sum += t;
    float avg_ms = sum / ITERS;
    float min_ms = *std::min_element(timings.begin(), timings.end());
    float max_ms = *std::max_element(timings.begin(), timings.end());
    
    double avg_tflops = (2.0 * M * N * K) / (avg_ms * 1e9);
    double min_tflops = (2.0 * M * N * K) / (max_ms * 1e9);
    double max_tflops = (2.0 * M * N * K) / (min_ms * 1e9);
    
    printf("Avg: %.3f ms (%.2f TFLOPS), Min: %.3f ms (%.2f TFLOPS), Max: %.3f ms (%.2f TFLOPS)\n", 
           avg_ms, avg_tflops, min_ms, max_tflops, max_ms, min_tflops);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template<int BM, int BN, int BK>
void dispatch_kernel(const char* kernel_name, half_t* d_A, half_t* d_B, float* d_C,
                    const std::vector<float>& h_C_cpu, int M, int N, int K, 
                    int lda, int ldb, int ldc, bool check_correctness) {
    
    auto run_kernel = [&](auto launch_func) {
        launch_func(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
        if (check_correctness) {
            std::vector<float> h_C_gpu(M * N);
            cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            bool passed = check_results(h_C_gpu.data(), h_C_cpu.data(), M, N, ldc);
            printf(passed ? "PASSED\n" : "FAILED\n");
            if (!passed) return false;
            cudaMemset(d_C, 0, M * N * sizeof(float));
        }
        benchmark_kernel(launch_func, d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
        return true;
    };
    
    if (strcmp(kernel_name, "cpasync") == 0) {
	if(BK == 32){ printf("Switch to cpasync_bk32\n"); return; };
        run_kernel([](half_t* dA, half_t* dB, float* dC, int m, int n, int k, int lda_, int ldb_, int ldc_) {
            gemm_cpasync_fixed::gemm_cpasync_launch<BM, BN, BK>(dA, dB, dC, m, n, k, lda_, ldb_, ldc_);
        });
    }
    else if (strcmp(kernel_name, "cpasync_bk32") == 0) {
        run_kernel([](half_t* dA, half_t* dB, float* dC, int m, int n, int k, int lda_, int ldb_, int ldc_) {
            gemm_cpasync_bk32::gemm_cpasync_launch<BM, BN, BK>(dA, dB, dC, m, n, k, lda_, ldb_, ldc_);
        });
    }
    else if (strcmp(kernel_name, "vector") == 0) {
        run_kernel([](half_t* dA, half_t* dB, float* dC, int m, int n, int k, int lda_, int ldb_, int ldc_) {
            gemm_vectorized_loads::gemm_vectorized_launch<BM, BN, BK>(dA, dB, dC, m, n, k, lda_, ldb_, ldc_);
        });
    }
    else if (strcmp(kernel_name, "scalar") == 0) {
        run_kernel([](half_t* dA, half_t* dB, float* dC, int m, int n, int k, int lda_, int ldb_, int ldc_) {
            gemm_scalar_loads::gemm_scalar_launch<BM, BN, BK>(dA, dB, dC, m, n, k, lda_, ldb_, ldc_);
        });
    }
    else if (strcmp(kernel_name, "pipelined") == 0) {
        run_kernel([](half_t* dA, half_t* dB, float* dC, int m, int n, int k, int lda_, int ldb_, int ldc_) {
            gemm_cpasync_pipelined::gemm_cpasync_pipelined_launch<BM, BN, BK, 3>(dA, dB, dC, m, n, k, lda_, ldb_, ldc_);
        });
    }
    else {
        fprintf(stderr, "Unknown kernel: %s\n", kernel_name);
    }
}

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s <kernel> <M> <N> <K> <BM> <BN> <BK> [--check]\n", argv[0]);
        fprintf(stderr, "Kernels: cpasync, cpasync_bk32, vector, scalar, pipelined\n");
        fprintf(stderr, "Example: %s cpasync 4096 4096 4096 128 64 32\n", argv[0]);
        return 1;
    }
    
    const char* kernel_name = argv[1];
    int M = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);
    int K = std::atoi(argv[4]);
    int BM = std::atoi(argv[5]);
    int BN = std::atoi(argv[6]);
    int BK = std::atoi(argv[7]);
    
    bool check_correctness = false;
    for (int i = 8; i < argc; i++) {
        if (strcmp(argv[i], "--check") == 0) {
            check_correctness = true;
            break;
        }
    }
    
    printf("Kernel: %s, Size: %dx%dx%d, Blocks: %dx%dx%d\n", 
           kernel_name, M, N, K, BM, BN, BK);
    
    int lda = K, ldb = K, ldc = N;
    
    std::vector<half_t> h_A(M * K), h_B(N * K);
    std::vector<float> h_C_cpu(M * N);
    initialize_matrices(h_A, h_B);
    
    if (check_correctness) {
        printf("Computing CPU reference...\n");
        cpu_gemm_reference(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K, lda, ldb, ldc);
    }
    
    half_t *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half_t));
    cudaMalloc(&d_B, N * K * sizeof(half_t));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(half_t), cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    printf("\ncuBLAS baseline: ");
    cudaMemset(d_C, 0, M * N * sizeof(float));
    const float alpha = 1.0f, beta = 0.0f;
    benchmark_kernel([&](half_t* dA, half_t* dB, float* dC, int m, int n, int k, int lda_, int ldb_, int ldc_) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
            &alpha, dB, CUDA_R_16F, n, dA, CUDA_R_16F, k, &beta, dC, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }, d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
    
    printf("Custom kernel: ");
    cudaMemset(d_C, 0, M * N * sizeof(float));
    
    if (BM == 128 && BN == 64 && BK == 32) {
        dispatch_kernel<128, 64, 32>(kernel_name, d_A, d_B, d_C, h_C_cpu, M, N, K, lda, ldb, ldc, check_correctness);
    } else if (BM == 128 && BN == 64 && BK == 64) {
        dispatch_kernel<128, 64, 64>(kernel_name, d_A, d_B, d_C, h_C_cpu, M, N, K, lda, ldb, ldc, check_correctness);
    } else if (BM == 64 && BN == 64 && BK == 32) {
        dispatch_kernel<64, 64, 32>(kernel_name, d_A, d_B, d_C, h_C_cpu, M, N, K, lda, ldb, ldc, check_correctness);
    } else if (BM == 64 && BN == 64 && BK == 64) {
        dispatch_kernel<64, 64, 64>(kernel_name, d_A, d_B, d_C, h_C_cpu, M, N, K, lda, ldb, ldc, check_correctness);
    } else if (BM == 128 && BN == 128 && BK == 64) {
        dispatch_kernel<128, 128, 64>(kernel_name, d_A, d_B, d_C, h_C_cpu, M, N, K, lda, ldb, ldc, check_correctness);
    } else {
        fprintf(stderr, "Unsupported block size combination.\n");
        return 1;
    }
    
    CHECK_CUBLAS(cublasDestroy(handle));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
