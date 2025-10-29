#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "gemm_universal.hpp"   // ← the header above

static void check(cudaError_t st, const char* msg) {
  if (st != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(st)); std::exit(1); }
}

int main(int argc, char** argv) {
  int M = 4096, N = 4096, K = 4096;
  if (argc > 1) M = std::atoi(argv[1]);
  if (argc > 2) N = std::atoi(argv[2]);
  if (argc > 3) K = std::atoi(argv[3]);
  float alpha = 1.0f, beta = 0.0f;

  printf("GEMM (Universal): M=%d N=%d K=%d  -> D = alpha*A*B + beta*C\n", M, N, K);

  // Leading dimensions consistent with chosen layouts
  int lda = (std::is_same<LayoutA, cutlass::layout::RowMajor>::value) ? K : M;
  int ldb = (std::is_same<LayoutB, cutlass::layout::RowMajor>::value) ? N : K;
  int ldc = (std::is_same<LayoutC, cutlass::layout::RowMajor>::value) ? N : M;
  int ldd = ldc;

  size_t bytesA = size_t(M) * K * sizeof(TypeA);
  size_t bytesB = size_t(K) * N * sizeof(TypeB);
  size_t bytesC = size_t(M) * N * sizeof(TypeC);
  size_t bytesD = size_t(M) * N * sizeof(TypeC);

  void *A_d=nullptr, *B_d=nullptr, *C_d=nullptr, *D_d=nullptr;
  check(cudaMalloc(&A_d, bytesA), "cudaMalloc A");
  check(cudaMalloc(&B_d, bytesB), "cudaMalloc B");
  check(cudaMalloc(&C_d, bytesC), "cudaMalloc C");
  check(cudaMalloc(&D_d, bytesD), "cudaMalloc D");
  check(cudaMemset(A_d, 0, bytesA), "memset A");
  check(cudaMemset(B_d, 0, bytesB), "memset B");
  check(cudaMemset(C_d, 0, bytesC), "memset C");
  check(cudaMemset(D_d, 0, bytesD), "memset D");

  // Problem & strides
  cutlass::gemm::GemmCoord problem(M, N, K);

  typename LayoutA::Stride strideA(lda);
  typename LayoutB::Stride strideB(ldb);
  typename LayoutC::Stride strideC(ldc);
  typename LayoutC::Stride strideD(ldd);

  // Batch strides (non-batched)
  int64_t bsA = 0, bsB = 0, bsC = 0, bsD = 0;

  EpilogueOp::Params ep{TypeAcc(alpha), TypeAcc(beta)};

  using Gemm = GemmConfigured;
  Gemm op;

  typename Gemm::Arguments args(
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem,
    /*batch_count*/ 1,
    ep,
    /*A,B,C,D*/ A_d, B_d, C_d, D_d,
    /*batch strides*/ bsA, bsB, bsC, bsD,
    /*layout-aware strides*/ strideA, strideB, strideC, strideD,
    /*gather/scatter*/ nullptr, nullptr, nullptr
  );

  // Split-K: serial if 1, parallel if >1 (compile-time fed)
  //   args.split_k_slices = SPLITK_SLICES;

  size_t workspace_bytes = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_bytes) {
    check(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc workspace");
  }

  cutlass::Status st = op.initialize(args, workspace);
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "initialize() failed: %d\n", int(st));
    return 1;
  }
  st = op();
  if (st != cutlass::Status::kSuccess) {
    fprintf(stderr, "run() failed: %d\n", int(st));
    return 1;
  }
  check(cudaDeviceSynchronize(), "sync");

  if (workspace) cudaFree(workspace);
  cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
  return 0;
}
