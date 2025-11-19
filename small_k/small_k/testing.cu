#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

//#include "no_eight.cuh"
#include "minimal2.cuh"
using namespace gemm_cute_min;

#define CHECK_CUDA(call) do {                               \
  cudaError_t err = call;                                   \
  if (err != cudaSuccess) {                                 \
    fprintf(stderr, "CUDA Error %s:%d: %s\n",               \
            __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(EXIT_FAILURE);                                     \
  }                                                         \
} while(0)

#define CHECK_CUBLAS(call) do {                             \
  cublasStatus_t status = call;                             \
  if (status != CUBLAS_STATUS_SUCCESS) {                    \
    fprintf(stderr, "cuBLAS Error %s:%d (status=%d)\n",     \
            __FILE__, __LINE__, int(status));               \
    exit(EXIT_FAILURE);                                     \
  }                                                         \
} while(0)

void fill_random(std::vector<float>& v) {
  for (auto &x : v) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
}

void compare_results(const std::vector<float>& ref,
                     const std::vector<float>& test,
                     int M, int N)
{
  const size_t n = static_cast<size_t>(M) * N;
  double max_abs = 0.0;
  double max_rel = 0.0;
  double ref_frob = 0.0, diff_frob = 0.0, ref_inf = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double r = static_cast<double>(ref[i]);
    double t = static_cast<double>(test[i]);
    double d = std::abs(r - t);
    max_abs = std::max(max_abs, d);
    ref_frob += r * r;
    diff_frob += d * d;
    ref_inf = std::max(ref_inf, std::abs(r));
  }

  double eps = 1e-7 * std::sqrt(static_cast<double>(n)) * std::max(1.0, ref_inf);
  double rel_frob = std::sqrt(diff_frob) / std::max(std::sqrt(ref_frob), eps);
  double stab = std::max(1e-6, 1e-6 * ref_inf);

  for (size_t i = 0; i < n; ++i) {
    double denom = std::max(static_cast<double>(std::abs(ref[i])), stab);
    double rel = std::abs(static_cast<double>(ref[i]) - static_cast<double>(test[i])) / denom;
    max_rel = std::max(max_rel, rel);
  }

  printf("Max abs error      = %.3e\n", max_abs);
  printf("Max elemwise rel   = %.3e  (stab=%.1e)\n", max_rel, stab);
  printf("Frobenius rel err  = %.3e\n", rel_frob);
}

int main(int argc, char** argv) {
  int M = 1000, N = 1000, K = 1000;
  if (argc == 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }

  printf("Testing GEMM %d x %d x %d\n", M, N, K);

  double flops = 2.0 * double(M) * double(N) * double(K);

  // Host allocations - FP32 for reference
  std::vector<float> hA_f32(size_t(M)*K);
  std::vector<float> hB_f32(size_t(K)*N);
  std::vector<float> hC(size_t(M)*N, 0.0f);
  std::vector<float> hC_ref(size_t(M)*N, 0.0f);
  
  fill_random(hA_f32);
  fill_random(hB_f32);

  // Host FP16 data - convert once on host
  std::vector<half_t> hA_f16(size_t(M)*K);
  std::vector<half_t> hB_f16_transposed(size_t(N)*K);  // Will be N×K after transpose
  
  // Convert A to FP16
  for (size_t i = 0; i < hA_f16.size(); ++i) {
    hA_f16[i] = half_t(hA_f32[i]);
  }
  
  // Convert and transpose B: K×N → N×K
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      hB_f16_transposed[n * K + k] = half_t(hB_f32[k * N + n]);
    }
  }

  // Device allocations - FP32 for reference, FP16 for custom kernel
  float *dA_f32 = nullptr, *dB_f32 = nullptr;
  half_t *dA_f16 = nullptr, *dB_f16 = nullptr;
  float *dC = nullptr, *dC_ref = nullptr;

  CHECK_CUDA(cudaMalloc(&dA_f32, size_t(M)*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB_f32, size_t(K)*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dA_f16, size_t(M)*K*sizeof(half_t)));
  CHECK_CUDA(cudaMalloc(&dB_f16, size_t(N)*K*sizeof(half_t)));  // N×K transposed
  CHECK_CUDA(cudaMalloc(&dC, size_t(M)*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC_ref, size_t(M)*N*sizeof(float)));

  // Copy to device - ONCE
  CHECK_CUDA(cudaMemcpy(dA_f32, hA_f32.data(), size_t(M)*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB_f32, hB_f32.data(), size_t(K)*N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dA_f16, hA_f16.data(), size_t(M)*K*sizeof(half_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB_f16, hB_f16_transposed.data(), size_t(N)*K*sizeof(half_t), cudaMemcpyHostToDevice));

  // Setup cuBLAS
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  const float alpha = 1.0f, beta = 0.0f;

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // ===========================================================================
  // CORRECTNESS TEST: FP32 cuBLAS vs Custom FP16
  // ===========================================================================
  printf("\n=== CORRECTNESS TEST (FP32 cuBLAS vs Custom FP16) ===\n");

  // FP32 reference GEMM (column-major cuBLAS)
  CHECK_CUBLAS(cublasSgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      dB_f32, N,  // FP32 B (K×N column-major)
      dA_f32, K,  // FP32 A (M×K column-major)
      &beta,
      dC_ref, N));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Custom FP16 GEMM (row-major)
  gemm_cute_min::gemm_cute_tc_fp16_launch(dA_f16, dB_f16, dC, M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Compare results
  CHECK_CUDA(cudaMemcpy(hC.data(), dC, size_t(M)*N*sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hC_ref.data(), dC_ref, size_t(M)*N*sizeof(float), cudaMemcpyDeviceToHost));
  
  printf("Correctness check:\n");
  compare_results(hC_ref, hC, M, N);

  printf("\nSample values (FP32 ref vs FP16 custom):\n");
  for (int i = 0; i < std::min(M, 4); ++i) {
    for (int j = 0; j < std::min(N, 4); ++j) {
      size_t idx = i * N + j;
      printf("C_ref[%d,%d]=% .6f  C[%d,%d]=% .6f  Δ=%+.2e\n",
             i, j, hC_ref[idx], i, j, hC[idx], double(hC_ref[idx] - hC[idx]));
    }
  }
  printf("...\n");

  // ===========================================================================
  // PERFORMANCE TEST: FP16 cuBLAS vs Custom FP16
  // ===========================================================================
  printf("\n=== PERFORMANCE TEST (FP16 cuBLAS vs Custom FP16) ===\n");

  // Reset outputs
  CHECK_CUDA(cudaMemset(dC, 0, size_t(M)*N*sizeof(float)));
  CHECK_CUDA(cudaMemset(dC_ref, 0, size_t(M)*N*sizeof(float)));

  // Warmup cuBLAS FP16
  printf("Warming up cuBLAS GemmEx (FP16)...\n");
  for (int i = 0; i < 3; ++i) {
    CHECK_CUBLAS(cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      dB_f32, CUDA_R_16F, N,  // Use FP32 data (cuBLAS converts internally)
      dA_f32, CUDA_R_16F, K,
      &beta,
      dC_ref, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F_FAST_16F,  // FP16 tensor cores
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed cuBLAS FP16
  const int num_iters = 10;
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_iters; ++i) {
    CHECK_CUBLAS(cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      dB_f32, CUDA_R_16F, N,
      dA_f32, CUDA_R_16F, K,
      &beta,
      dC_ref, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F_FAST_16F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float time_cublas = 0;
  CHECK_CUDA(cudaEventElapsedTime(&time_cublas, start, stop));
  time_cublas /= num_iters;
  double gflops_cublas = flops / (time_cublas * 1e-3) / 1e9;
  printf("cuBLAS GemmEx (FP16 TC): %.3f ms, %.2f GFLOPS\n", time_cublas, gflops_cublas);

  // Warmup custom GEMM
  for (int i = 0; i < 3; ++i)
    gemm_cute_min::gemm_cute_tc_fp16_launch(dA_f16, dB_f16, dC, M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());
  
  CHECK_CUDA(cudaMemset(dC, 0, size_t(M)*N*sizeof(float)));

  // Timed custom GEMM - NO conversion overhead!
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_iters; ++i) {
    gemm_cute_min::gemm_cute_tc_fp16_launch(dA_f16, dB_f16, dC, M, N, K);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float time_custom = 0;
  CHECK_CUDA(cudaEventElapsedTime(&time_custom, start, stop));
  time_custom /= num_iters;
  double gflops_custom = flops / (time_custom * 1e-3) / 1e9;
  printf("Custom GEMM (FP16 TC):   %.3f ms, %.2f GFLOPS\n", time_custom, gflops_custom);
  printf("Speedup vs cuBLAS:       %.2fx\n", time_cublas / time_custom);
  printf("Efficiency:              %.1f%% of cuBLAS\n", 100.0 * time_cublas / time_custom);

  // Cleanup
  CHECK_CUDA(cudaFree(dA_f32));
  CHECK_CUDA(cudaFree(dB_f32));
  CHECK_CUDA(cudaFree(dA_f16));
  CHECK_CUDA(cudaFree(dB_f16));
  CHECK_CUDA(cudaFree(dC));
  CHECK_CUDA(cudaFree(dC_ref));
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  printf("\nDone.\n");
  return 0;
}
