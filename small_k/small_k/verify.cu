#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <tuple>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "minimal.cuh"  // defines gemm_cute_min::gemm_cute_tc_fp16_launch, half_t
//#include "no_eight.cuh"
using namespace gemm_cute_min;

// --------------------------------- utils -------------------------------------
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

// ---------------------- CPU reference: row-major TN --------------------------
// C_out = alpha * (A[M,K] * B_T[N,K]^T) + beta * C0[M,N]
// A is row-major MxK, B_T is row-major NxK (i.e., transposed KxN), C_out/C0 row-major MxN
static void gemm_cpu_ref_rowmajor_TN(
    float* C_out,          int ldC,   // = N
    const float* A,        int ldA,   // = K
    const float* B_T,      int ldB_T, // = K
    const float* C0,                 // MxN row-major (input)
    int M, int N, int K,
    float alpha, float beta)
{
  // Initialize C_out from beta*C0
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C_out[i*ldC + j] = beta * C0[i*ldC + j];
    }
  }

  // Accumulate alpha*(A*B^T)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.f;
      // B_T is NxK row-major => B[k,j] = B_T[j*ldB_T + k]
      for (int k = 0; k < K; ++k) {
        float a = A[i*ldA + k];
        float b = B_T[j*ldB_T + k];
        sum += a * b;
      }
      C_out[i*ldC + j] += alpha * sum;
    }
  }
}

// ---------------------------- verification -----------------------------------
static bool verify_custom_kernel(int M, int N, int K, float alpha, float beta) {
  printf("Verifying custom kernel: %d x %d x %d (alpha=%.3f, beta=%.3f)\n",
         M, N, K, alpha, beta);

  // Host matrices
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

  // Device allocations
  half_t *dA_f16 = nullptr, *dB_f16 = nullptr;
  float  *dC     = nullptr;

  CHECK_CUDA(cudaMalloc(&dA_f16, size_t(M)*K * sizeof(half_t)));
  CHECK_CUDA(cudaMalloc(&dB_f16, size_t(N)*K * sizeof(half_t)));
  CHECK_CUDA(cudaMalloc(&dC,     size_t(M)*N * sizeof(float)));

  // Convert A,B_T to FP16
  std::vector<half_t> hA_f16(size_t(M)*K);
  std::vector<half_t> hB_f16(size_t(N)*K);
  for (size_t i = 0; i < hA_f16.size(); ++i) hA_f16[i] = half_t(hA[i]);
  for (size_t i = 0; i < hB_f16.size(); ++i) hB_f16[i] = half_t(hB_T[i]);

  CHECK_CUDA(cudaMemcpy(dA_f16, hA_f16.data(), size_t(M)*K * sizeof(half_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB_f16, hB_f16.data(), size_t(N)*K * sizeof(half_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC, 0, size_t(M)*N * sizeof(float)));

  // Run custom kernel
  gemm_cute_tc_fp16_launch(dA_f16, dB_f16, dC, M, N, K);
  CHECK_CUDA(cudaGetLastError()); 
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy D back
  CHECK_CUDA(cudaMemcpy(hD.data(), dC, size_t(M)*N * sizeof(float), cudaMemcpyDeviceToHost));

  // Compose C_gpu = alpha * D + beta * C0
  for (int i = 0; i < M*N; ++i) {
    hC_gpu[i] = alpha * hD[i] + beta * hC0[i];
  }

  // CPU reference
  gemm_cpu_ref_rowmajor_TN(hC_cpu.data(), /*ldC=*/N,
                           hA.data(),     /*ldA=*/K,
                           hB_T.data(),   /*ldB_T=*/K,
                           hC0.data(),
                           M, N, K,
                           alpha, beta);

  // ========== PRINT MATRICES FOR SMALL SIZES ==========
  const bool print_full = (M <= 16 && N <= 16);
  
  if (print_full) {
    printf("\n--- Input A (MxK = %dx%d, row-major) ---\n", M, K);
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        printf("%7.3f ", hA[i*K + k]);
      }
      printf("\n");
    }
    
    printf("\n--- Input B_T (NxK = %dx%d, row-major) ---\n", N, K);
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        printf("%7.3f ", hB_T[n*K + k]);
      }
      printf("\n");
    }
    
    printf("\n--- GPU Output (raw D, before alpha/beta) ---\n");
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        printf("%10.4f ", hD[i*N + j]);
      }
      printf("\n");
    }
    
    printf("\n--- CPU Reference (C_cpu) ---\n");
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        printf("%10.4f ", hC_cpu[i*N + j]);
      }
      printf("\n");
    }
    
    printf("\n--- GPU Result (C_gpu = alpha*D + beta*C0) ---\n");
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        printf("%10.4f ", hC_gpu[i*N + j]);
      }
      printf("\n");
    }
  }

  // ========== COMPARE AND REPORT MISMATCHES ==========
  int mismatches = 0;
  const int kMaxReport = print_full ? M*N : 100;  // Report all for small, limit for large
  double max_abs_err = 0.0, max_rel_err = 0.0;
  const double rtol = 1e-2;
  const double atol = 1e-3;

  std::vector<std::tuple<int,int,float,float,double,double>> errors;

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int idx = i*N + j;
      float got = hC_gpu[idx];
      float ref = hC_cpu[idx];

      double abs_err = std::fabs(double(got) - double(ref));
      double rel_err = abs_err / (std::fabs(double(ref)) + 1e-30);

      if (abs_err > max_abs_err) max_abs_err = abs_err;
      if (rel_err > max_rel_err) max_rel_err = rel_err;

      if (!within_tol(got, ref, rtol, atol)) {
        errors.push_back({i, j, got, ref, abs_err, rel_err});
        ++mismatches;
      }
    }
  }

// Print mismatches with analysis
if (!errors.empty()) {
  // Analyze zero pattern
  int zero_count = 0;
  int min_zero_row = M, max_zero_row = -1;
  int min_zero_col = N, max_zero_col = -1;
  
  for (auto& [i, j, got, ref, abs_err, rel_err] : errors) {
    if (got == 0.0f) {
      zero_count++;
      if (i < min_zero_row) min_zero_row = i;
      if (i > max_zero_row) max_zero_row = i;
      if (j < min_zero_col) min_zero_col = j;
      if (j > max_zero_col) max_zero_col = j;
    }
  }
  
  printf("\n=== MISMATCH ANALYSIS ===\n");
  printf("Total mismatches: %d out of %d (%.3f%%)\n", 
         (int)errors.size(), M*N, 100.0*errors.size()/(M*N));
  printf("Zero mismatches: %d (%.1f%% of mismatches)\n", 
         zero_count, 100.0*zero_count/errors.size());
  
  if (zero_count > 0) {
    printf("Zero error range: rows [%d, %d], cols [%d, %d]\n",
           min_zero_row, max_zero_row, min_zero_col, max_zero_col);
    
    printf("\n--- ZERO MISMATCHES (first 50) ---\n");
    int printed = 0;
    for (auto& [i, j, got, ref, abs_err, rel_err] : errors) {
      if (got == 0.0f && printed < 50) {
        printf("  [%4d,%4d]: got=%10.6f  ref=%10.6f  abs=%.3e  rel=%.3e\n",
               i, j, got, ref, abs_err, rel_err);
        printed++;
      }
    }
    if (zero_count > 50) {
      printf("  ... and %d more zeros\n", zero_count - 50);
    }
  }
  
  printf("\n--- NON-ZERO MISMATCHES (first 20) ---\n");
  int printed = 0;
  for (auto& [i, j, got, ref, abs_err, rel_err] : errors) {
    if (got != 0.0f && printed < 20) {
      printf("  [%4d,%4d]: got=%10.6f  ref=%10.6f  abs=%.3e  rel=%.3e\n",
             i, j, got, ref, abs_err, rel_err);
      printed++;
    }
  }
  
  int non_zero_count = errors.size() - zero_count;
  if (non_zero_count > 20) {
    printf("  ... and %d more non-zero errors\n", non_zero_count - 20);
  }
}
  // Cleanup
  cudaFree(dA_f16);
  cudaFree(dB_f16);
  cudaFree(dC);

  printf("\nMax abs error: %.3e\n", max_abs_err);
  printf("Max rel error: %.3e\n", max_rel_err);

  if (mismatches) {
    printf("FAILED: %d mismatches out of %lld (%.3f%%)\n",
           mismatches, (long long)M*N, 100.0*mismatches/(M*N));
    return false;
  }
  printf("PASSED: within tolerance\n\n");
  return true;
}
// ----------------------------------- main ------------------------------------
int main(int argc, char** argv) {

  int M=2, N=3, K=4;
  std::vector<float> A = {
    1,2,3,4,
    5,6,7,8,    // row-major 2x4
  };
  std::vector<float> B_orig = {
    // KxN = 4x3, column-major view aside, but we store row-major KxN
    1,  10, 100,
    2,  20, 200,
    3,  30, 300,
    4,  40, 400
  };
  // Build B_T (N×K)
  std::vector<float> B_T(N*K);
  for (int k=0;k<K;++k) for (int n=0;n<N;++n) B_T[n*K+k] = B_orig[k*N+n];



  if (argc == 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }

  printf("=== Custom Kernel Verification ===\n\n");

  // These now work because we post-process alpha/beta on the host:
  if (!verify_custom_kernel(M, N, K, 1.0f, 0.0f)) return 1;
  if (!verify_custom_kernel(M, N, K, 1.0f, 1.0f)) return 1;
  if (!verify_custom_kernel(M, N, K, 2.0f, 0.5f)) return 1;

  printf("=== Testing various sizes ===\n");
  std::vector<std::tuple<int,int,int>> test_sizes = {
      {64, 64, 64},
      {128, 128, 128},
      {256, 256, 256},
      {512, 512, 512},
      {1024, 1024, 1024},
      {1000, 1000, 1000},
  };

  for (auto [m,n,k] : test_sizes) {
    if (!verify_custom_kernel(m, n, k, 1.0f, 0.0f)) {
      printf("Failed at size %dx%dx%d\n", m,n,k);
      return 1;
    }
  }

  printf("\n=== ALL TESTS PASSED ===\n");
  return 0;
}

