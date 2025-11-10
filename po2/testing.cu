#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

//#include "gemm_cpasync2.h"
#include "gemm_cpasync.h"
//#include "gemm_fixed_layout.h"
//#include "gemm_persistent_splitk.cuh"
//#include "cute_gemm_simple.h"

using cutlass::half_t;

// CPU reference GEMM
void cpu_gemm_reference(const half_t* A, const half_t* B, float* C,
                        int M, int N, int K,
                        int lda, int ldb, int ldc)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a_val = static_cast<float>(A[m * lda + k]);
                float b_val = static_cast<float>(B[n * ldb + k]); // B is [N,K]
                sum += a_val * b_val;
            }
            C[m * ldc + n] = sum;
        }
    }
}

// Result check
bool check_results(const float* C_gpu, const float* C_cpu,
                   int M, int N, int ldc,
                   float rel_tol = 1e-2f, float abs_tol = 1e-3f)
{
    bool passed = true;
    int num_errors = 0;
    int max_errors_to_print = 20;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int idx = m * ldc + n;
            float gpu_val = C_gpu[idx];
            float cpu_val = C_cpu[idx];
            float diff = std::abs(gpu_val - cpu_val);
            float rel_diff = diff / (std::abs(cpu_val) + 1e-8f);

            if (diff > abs_tol && rel_diff > rel_tol) {
                if (num_errors < max_errors_to_print) {
                    std::cout << "  Mismatch at [" << m << "," << n << "]: "
                              << "GPU=" << gpu_val << ", CPU=" << cpu_val
                              << ", diff=" << diff << ", rel_diff=" << rel_diff << std::endl;
                }
                ++num_errors; passed = false;
            }
        }
    }
    if (num_errors) {
        std::cout << "  Total errors: " << num_errors
                  << " out of " << (M * N) << std::endl;
    }
    return passed;
}

void initialize_matrices(std::vector<half_t>& A, std::vector<half_t>& B,
                         int M, int N, int K)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& v : A) v = half_t(dis(gen));
    for (auto& v : B) v = half_t(dis(gen));
}

template<typename Func>
void test_kernel(const char* name, Func kernel_func,
                 half_t* d_A, half_t* d_B, float* d_C,
                 const std::vector<half_t>& h_A,
                 const std::vector<half_t>& h_B,
                 const std::vector<float>& h_C_cpu,
                 int M, int N, int K, int lda, int ldb, int ldc)
{
    std::cout << "\n========================================\n";
    std::cout << "Testing: " << name << "\n";
    std::cout << "========================================\n";

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(half_t), cudaMemcpyHostToDevice);
    // Note: kernel_func is responsible for zeroing d_C if needed

    kernel_func(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "✗ CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    cudaDeviceSynchronize();

    std::vector<float> h_C_gpu(M * N);
    cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Checking results...\n";
    bool passed = check_results(h_C_gpu.data(), h_C_cpu.data(), M, N, ldc);
    if (passed) std::cout << "✓ PASSED!\n";
    else        std::cout << "✗ FAILED!\n";

    if (!passed) return;

    // Timing
    const int num_warmup = 5, num_iters = 100;
    for (int i = 0; i < num_warmup; ++i) kernel_func(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < num_iters; ++i) kernel_func(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    float avg = ms / num_iters;
    double gflops = (2.0 * M * N * K * 1e-9) / (avg * 1e-3);
    std::cout << "Performance: " << avg << " ms, " << gflops << " GFLOPS\n";
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char** argv)
{
    int M = 256, N = 256, K = 256;
    if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";

    int lda = K, ldb = K, ldc = N;

    std::vector<half_t> h_A(M * K), h_B(N * K);
    std::vector<float>  h_C_cpu(M * N);
    std::cout << "Initializing matrices...\n";
    initialize_matrices(h_A, h_B, M, N, K);
    std::cout << "Computing CPU reference...\n";
    cpu_gemm_reference(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K, lda, ldb, ldc);

    half_t *d_A, *d_B; float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half_t));
    cudaMalloc(&d_B, N * K * sizeof(half_t));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Test 1: cp.async Single-Stage (needs explicit memset)
    test_kernel("cp.async Single-Stage (Fixed Sync)",
        [](half_t* dA, half_t* dB, float* dC, int m, int n, int k, int lda_, int ldb_, int ldc_) {
            cudaMemset(dC, 0, m * n * sizeof(float));
            gemm_cpasync_fixed::gemm_cpasync_launch<128, 64, 64>(dA, dB, dC, m, n, k, lda_, ldb_, ldc_);
        }, d_A, d_B, d_C, h_A, h_B, h_C_cpu, M, N, K, lda, ldb, ldc);

/*    test_kernel("double buffer",
        [](half_t* dA, half_t* dB, float* dC, int m, int n, int k, int lda_, int ldb_, int ldc_) {
            int split_k = 8;
            gemm_cute_doublebuf::gemm_cute_doublebuf_launch<128, 64, 64>(
                dA, dB, dC, m, n, k, lda_, ldb_, ldc_, split_k);
        }, d_A, d_B, d_C, h_A, h_B, h_C_cpu, M, N, K, lda, ldb, ldc);
  */  
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
