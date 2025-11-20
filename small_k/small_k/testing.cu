#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "minimal.cuh"
using namespace gemm_cute_min;


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

int main(int argc, char** argv) {
  int M = 1000, N = 1000, K = 1000;
  if (argc == 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }

  printf("Testing GEMM %d x %d x %d\n", M, N, K);

  double flops = 2.0 * double(M) * double(N) * double(K);

  // Host allocations
  std::vector<float> hA_f32(size_t(M)*K);
  std::vector<float> hB_f32(size_t(K)*N);
  
  fill_random(hA_f32);
  fill_random(hB_f32);

  // Convert to FP16
  std::vector<half_t> hA_f16(size_t(M)*K);
  std::vector<half_t> hB_f16_transposed(size_t(N)*K);
  
  for (size_t i = 0; i < hA_f16.size(); ++i) {
    hA_f16[i] = half_t(hA_f32[i]);
  }
  
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      hB_f16_transposed[n * K + k] = half_t(hB_f32[k * N + n]);
    }
  }

  // Device allocations
  float *dA_f32 = nullptr, *dB_f32 = nullptr;
  half_t *dA_f16 = nullptr, *dB_f16 = nullptr;
  float *dC = nullptr, *dC_ref = nullptr;

  CHECK_CUDA(cudaMalloc(&dA_f32, size_t(M)*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB_f32, size_t(K)*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dA_f16, size_t(M)*K*sizeof(half_t)));
  CHECK_CUDA(cudaMalloc(&dB_f16, size_t(N)*K*sizeof(half_t)));
  CHECK_CUDA(cudaMalloc(&dC, size_t(M)*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC_ref, size_t(M)*N*sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dA_f32, hA_f32.data(), size_t(M)*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB_f32, hB_f32.data(), size_t(K)*N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dA_f16, hA_f16.data(), size_t(M)*K*sizeof(half_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB_f16, hB_f16_transposed.data(), size_t(N)*K*sizeof(half_t), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  const float alpha = 1.0f, beta = 0.0f;

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Warmup cuBLAS FP16
  for (int i = 0; i < 3; ++i) {
    CHECK_CUBLAS(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
      &alpha, dB_f32, CUDA_R_16F, N, dA_f32, CUDA_R_16F, K,
      &beta, dC_ref, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark cuBLAS FP16
  const int num_iters = 10;
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_iters; ++i) {
    CHECK_CUBLAS(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
      &alpha, dB_f32, CUDA_R_16F, N, dA_f32, CUDA_R_16F, K,
      &beta, dC_ref, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float time_cublas = 0;
  CHECK_CUDA(cudaEventElapsedTime(&time_cublas, start, stop));
  time_cublas /= num_iters;
  double gflops_cublas = flops / (time_cublas * 1e-3) / 1e9;
  printf("cuBLAS (FP16 TC): %.3f ms, %.2f GFLOPS\n", time_cublas, gflops_cublas);

  // Warmup custom GEMM
  for (int i = 0; i < 3; ++i)
    gemm_cute_min::gemm_cute_tc_fp16_launch(dA_f16, dB_f16, dC, M, N, K);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Benchmark custom GEMM
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
  printf("Custom (FP16 TC): %.3f ms, %.2f GFLOPS\n", time_custom, gflops_custom);
  printf("Speedup:          %.2fx\n", time_cublas / time_custom);

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

  return 0;
}
