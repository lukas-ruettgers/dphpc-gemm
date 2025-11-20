#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <tuple>
#include <cuda_runtime.h>

#include "minimal.cuh"
using namespace gemm_cute_min;

#define CHECK_CUDA(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

static inline bool within_tol(float got, float ref, double rtol, double atol) {
  double diff = std::fabs(double(got) - double(ref));
  double tol  = atol + rtol * std::fabs(double(ref));
  return diff <= tol;
}

static inline void fill_random(std::vector<float>& v, unsigned int seed) {
  srand(seed);
  for (auto &x : v) x = float(rand())/RAND_MAX - 0.5f;
}

// CPU reference: C_out = alpha * (A[M,K] * B_T[N,K]^T) + beta * C0[M,N]
static void gemm_cpu_ref_rowmajor_TN(
    float* C_out,          int ldC,
    const float* A,        int ldA,
    const float* B_T,      int ldB_T,
    const float* C0,
    int M, int N, int K,
    float alpha, float beta)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C_out[i*ldC + j] = beta * C0[i*ldC + j];
    }
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.f;
      for (int k = 0; k < K; ++k) {
        sum += A[i*ldA + k] * B_T[j*ldB_T + k];
      }
      C_out[i*ldC + j] += alpha * sum;
    }
  }
}

static bool verify_custom_kernel(int M, int N, int K, float alpha, float beta) {
  printf("Verifying: %dx%dx%d (alpha=%.3f, beta=%.3f)\n", M, N, K, alpha, beta);

  std::vector<float> hA(size_t(M)*K);
  std::vector<float> hB_orig(size_t(K)*N);
  std::vector<float> hB_T(size_t(N)*K);
  std::vector<float> hC0(size_t(M)*N);
  std::vector<float> hD(size_t(M)*N);
  std::vector<float> hC_cpu(size_t(M)*N);
  std::vector<float> hC_gpu(size_t(M)*N);

  fill_random(hA,      1234);
  fill_random(hB_orig, 5678);
  fill_random(hC0,     9012);

  // Transpose B: KxN -> NxK
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      hB_T[n*K + k] = hB_orig[k*N + n];

  half_t *dA_f16 = nullptr, *dB_f16 = nullptr;
  float  *dC     = nullptr;

  CHECK_CUDA(cudaMalloc(&dA_f16, size_t(M)*K * sizeof(half_t)));
  CHECK_CUDA(cudaMalloc(&dB_f16, size_t(N)*K * sizeof(half_t)));
  CHECK_CUDA(cudaMalloc(&dC,     size_t(M)*N * sizeof(float)));

  std::vector<half_t> hA_f16(size_t(M)*K);
  std::vector<half_t> hB_f16(size_t(N)*K);
  for (size_t i = 0; i < hA_f16.size(); ++i) hA_f16[i] = half_t(hA[i]);
  for (size_t i = 0; i < hB_f16.size(); ++i) hB_f16[i] = half_t(hB_T[i]);

  CHECK_CUDA(cudaMemcpy(dA_f16, hA_f16.data(), size_t(M)*K * sizeof(half_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB_f16, hB_f16.data(), size_t(N)*K * sizeof(half_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC, 0, size_t(M)*N * sizeof(float)));

  gemm_cute_tc_fp16_launch(dA_f16, dB_f16, dC, M, N, K);
  CHECK_CUDA(cudaGetLastError()); 
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(hD.data(), dC, size_t(M)*N * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < M*N; ++i) {
    hC_gpu[i] = alpha * hD[i] + beta * hC0[i];
  }

  gemm_cpu_ref_rowmajor_TN(hC_cpu.data(), N, hA.data(), K, hB_T.data(), K,
                           hC0.data(), M, N, K, alpha, beta);

  const bool print_full = (M <= 16 && N <= 16);
  
  if (print_full) {
    printf("\n--- Input A (MxK = %dx%d) ---\n", M, K);
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) printf("%7.3f ", hA[i*K + k]);
      printf("\n");
    }
    
    printf("\n--- Input B_T (NxK = %dx%d) ---\n", N, K);
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) printf("%7.3f ", hB_T[n*K + k]);
      printf("\n");
    }
    
    printf("\n--- GPU Output (raw D) ---\n");
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) printf("%10.4f ", hD[i*N + j]);
      printf("\n");
    }
    
    printf("\n--- CPU Reference ---\n");
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) printf("%10.4f ", hC_cpu[i*N + j]);
      printf("\n");
    }
    
    printf("\n--- GPU Result (alpha*D + beta*C0) ---\n");
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) printf("%10.4f ", hC_gpu[i*N + j]);
      printf("\n");
    }
  }

  int mismatches = 0;
  const double rtol = 1e-2, atol = 1e-3;

  std::vector<std::tuple<int,int,float,float>> errors;

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int idx = i*N + j;
      float got = hC_gpu[idx];
      float ref = hC_cpu[idx];

      if (!within_tol(got, ref, rtol, atol)) {
        errors.push_back({i, j, got, ref});
        ++mismatches;
      }
    }
  }

  if (!errors.empty()) {
    printf("\n=== MISMATCHES: %d out of %d (%.3f%%) ===\n", 
           (int)errors.size(), M*N, 100.0*errors.size()/(M*N));
    
    int printed = 0;
    for (auto& [i, j, got, ref] : errors) {
      if (printed++ < 20) {
        printf("  [%4d,%4d]: got=%10.6f  ref=%10.6f\n", i, j, got, ref);
      }
    }
    if (errors.size() > 20) printf("  ... and %d more\n", (int)errors.size() - 20);
  }

  cudaFree(dA_f16);
  cudaFree(dB_f16);
  cudaFree(dC);

  if (mismatches) {
    printf("FAILED: %d mismatches\n\n", mismatches);
    return false;
  }
  printf("PASSED\n\n");
  return true;
}

int main(int argc, char** argv) {
  int M = 2, N = 3, K = 4;

  if (argc == 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }

  if (!verify_custom_kernel(M, N, K, 1.0f, 0.0f)) return 1;
  if (!verify_custom_kernel(M, N, K, 1.0f, 1.0f)) return 1;
  if (!verify_custom_kernel(M, N, K, 2.0f, 0.5f)) return 1;

  printf("ALL TESTS PASSED\n");
  return 0;
}
