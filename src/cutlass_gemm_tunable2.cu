#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/layout/matrix.h"

#include "utils.hpp"

#include <cuda_runtime.h>

static void checkCuda(cudaError_t res, const char* msg = "") {
  if (res != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(res) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}


///////////////////////////////////////////////////////////////////////////////////////////////////

// Helper to allocate and initialize host/device memory
template <typename T>
T *alloc_device(size_t count) {
  T *dptr = nullptr;
  checkCuda(cudaMalloc((void**)&dptr, count * sizeof(T)), "cudaMalloc");
  return dptr;
}

template <typename T>
T *alloc_host(size_t count) {
  T *h = (T*)malloc(sizeof(T)*count);
  if (!h) { std::cerr<<"host malloc failed\n"; std::exit(EXIT_FAILURE); }
  return h;
}

// Fill with simple pattern
template <typename T>
void fill_random(T* ptr, size_t n) {
  for (size_t i=0;i<n;i++) {
    float v = (float)( (rand() % 100) - 50 ) / 23.0f;
    ptr[i] = (T)v;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int TB_M, int TB_N, int TB_K,
  int WP_M, int WP_N, int WP_K,
  int INST_M, int INST_N, int INST_K
>
cudaError_t CutlassMgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  cutlass::half_t const *A,
  int lda,
  cutlass::half_t const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  int iters){
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, 
        cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,
        cutlass::gemm::GemmShape<WP_M, WP_N, WP_K>,
        cutlass::gemm::GemmShape<INST_M, INST_N, INST_K>, // instruction (NEW)
        cutlass::epilogue::thread::LinearCombination<
            ElementC,
            1,
            ElementAccumulator,
            ElementAccumulator
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>; // default swizzle


    typename Gemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                        {A, lda},    // Tensor-ref for source matrix A
                        {B, ldb},    // Tensor-ref for source matrix B
                        {C, ldc},    // Tensor-ref for source matrix C
                        {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                        {alpha, beta}); // Scalars used in the Epilogue

    Gemm gemm_op;
    gemm_op.initialize(args);
    // CUDA_CHECK_MSG_RET(gemm_op.initialize(args), "Gemm initialize failed: ")


    double t_ms = time_cuda_event([&]() { gemm_op(); }, 3, iters); //handles warmup
    double tflops = 2.0 * M * N * K / (t_ms * 1e9);

    std::cout << "Kernel: TB=" << TB_M << "x" << TB_N << "x" << TB_K
              << " WP=" << WP_M << "x" << WP_N << "x" << WP_K
              << " INST=" << INST_M << "x" << INST_N << "x" << INST_K
              << " CUTLASS: M=" << M << " N=" << N << " K=" << K
              << "  Time=" << t_ms << " ms  Perf=" << tflops << " TFLOPs\n";

    return cudaSuccess;
}

// -----------------------------------------------------------------------------
// Dispatch table: map runtime tile choice -> precompiled template call
// -----------------------------------------------------------------------------

cudaError_t run_cutlass_dispatch(
    int tbA, int tbB, int tbC,
    int wpA, int wpB, int wpC,
    int instA, int instB, int instC,
    int M, int N, int K,
    float alpha,
    cutlass::half_t const *dA, int lda,
    cutlass::half_t const *dB, int ldb,
    float *dC, int ldc,
    float beta,
    int iters)
{
  // NOTE: Add or remove cases as you compile more instantiations.
  // Keep these in the order you want tested (common -> exotic).

  //128x256x32 64x64x32 8x8x4
  if (tbA == 128 && tbB == 256 && tbC == 32 &&
      wpA == 64  && wpB == 64  && wpC == 32 &&
      instA == 8  && instB == 8  && instC == 4
      ) {

    return CutlassMgemmNN<128,256,32,64,64,32, 8,8,4>(
        M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc, iters);
  }

  //128x256x32 64x64x32 16x8x8
  if (tbA == 128 && tbB == 256 && tbC == 32 &&
      wpA == 64  && wpB == 64  && wpC == 32 &&
      instA == 16  && instB == 8  && instC == 8
      ) {

    return CutlassMgemmNN<128,256,32,64,64,32, 16,8,8>(
        M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc, iters);
  }

  //128x128x32 64x64x32 8x8x4
  if (tbA == 128 && tbB == 128 && tbC == 32 &&
      wpA == 64  && wpB == 64  && wpC == 32 &&
      instA == 8  && instB == 8  && instC == 4
      ) {

    return CutlassMgemmNN<128,128,32,64,64,32, 8,8,4>(
        M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc, iters);
  }

  //128x128x32 64x64x32 16x8x8
  if (tbA == 128 && tbB == 128 && tbC == 32 &&
      wpA == 64  && wpB == 64  && wpC == 32 &&
      instA == 16  && instB == 8  && instC == 8
      ) {

    return CutlassMgemmNN<128,128,32,64,64,32, 16,8,8>(
        M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc, iters);
  }

  //256x128x32 64x64x32 8x8x4
  if (tbA == 256 && tbB == 128 && tbC == 32 &&
      wpA == 64  && wpB == 64  && wpC == 32 &&
      instA == 8  && instB == 8  && instC == 4
      ) {

    return CutlassMgemmNN<256,128,32,64,64,32,8,8,4>(
        M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc, iters);
  }

  //256x128x32 64x64x32 16x8x8
  if (tbA == 256 && tbB == 128 && tbC == 32 &&
      wpA == 64  && wpB == 64  && wpC == 32 &&
      instA == 16  && instB == 8  && instC == 8
      ) {

    return CutlassMgemmNN<256,128,32,64,64,32, 16,8,8>(
        M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc, iters);
  }

  
  // // fallback: unsupported
  std::cerr << "run_cutlass_dispatch: unsupported tile/warp configuration: "
            << tbA << "x" << tbB << "x" << tbC << " TB, "
            << wpA << "x" << wpB << "x" << wpC << " WP\n"
            << instA << "x" << instB << "x" << instC << " WP\n";
  std::cerr << "Add a matching CutlassMgemmNN<...> instantiation to the binary.\n";
  return cudaErrorInvalidValue;
}



// Simple CLI parsing helper.
struct Args {
  int M = 4096, N = 4096, K = 32;
  std::string threadblock = "128x256x32";
  std::string warp = "64x64x32";
  std::string inst = "8x8x4";
  int iters = 10;
  float alpha = 1.0f, beta = 0.0f;
};

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i=1;i<argc;i++){
    std::string a(argv[i]);
    if (a=="--M") args.M = atoi(argv[++i]);
    else if (a=="--N") args.N = atoi(argv[++i]);
    else if (a=="--K") args.K = atoi(argv[++i]);
    else if (a=="--threadblock") args.threadblock = argv[++i];
    else if (a=="--warp") args.warp = argv[++i];
    else if (a=="--inst") args.inst = argv[++i];
    else if (a=="--iters") args.iters = atoi(argv[++i]);
    else if (a=="--alpha") args.alpha = atof(argv[++i]);
    else if (a=="--beta") args.beta = atof(argv[++i]);
    else {
      std::cerr << "Unknown arg: " << a << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  return args;
}

// parse "AxBxC" -> tuple
void parse_shape(const std::string &s, int &A, int &B, int &C) {
  int a=0,b=0,c=0;
  if (sscanf(s.c_str(), "%dx%dx%d", &a, &b, &c) != 3) {
    std::cerr << "Bad shape string: " << s << ". Expected AxBxC\n";
    std::exit(EXIT_FAILURE);
  }
  A=a; B=b; C=c;
}

int main(int argc, char **argv){
  cudaError_t result;

  Args args = parse_args(argc, argv);

  int tbA, tbB, tbC, wpA, wpB, wpC, instA, instB, instC;
  parse_shape(args.threadblock, tbA, tbB, tbC);
  parse_shape(args.warp, wpA, wpB, wpC);
  parse_shape(args.inst, instA, instB, instC);
  
  // Compute leading dimensions for each matrix.
  int lda = args.K;
  int ldb = args.N;
  int ldc = args.N;
  size_t size_A = size_t(args.M) * size_t(args.K);
  size_t size_B = size_t(args.K) * size_t(args.N);
  size_t size_C = size_t(args.M) * size_t(args.N);


  cutlass::half_t *hA = alloc_host<cutlass::half_t>(size_A);
  cutlass::half_t *hB = alloc_host<cutlass::half_t>(size_B);
  float *hC = alloc_host<float>(size_C);
 
  fill_random<cutlass::half_t>(hA, size_A);
  fill_random<cutlass::half_t>(hB, size_B);
  for (size_t i=0;i<size_C;i++) hC[i] = 0.0f;

  cutlass::half_t *dA = alloc_device<cutlass::half_t>(size_A);
  cutlass::half_t *dB = alloc_device<cutlass::half_t>(size_B);
  float *dC = alloc_device<float>(size_C);

  checkCuda(cudaMemcpy(dA, hA, size_A*sizeof(cutlass::half_t), cudaMemcpyHostToDevice), "memcpy A");
  checkCuda(cudaMemcpy(dB, hB, size_B*sizeof(cutlass::half_t), cudaMemcpyHostToDevice), "memcpy B");
  checkCuda(cudaMemcpy(dC, hC, size_C*sizeof(float), cudaMemcpyHostToDevice), "memcpy C");

  //
  // Launch CUTLASS GEMM.
  //

  // Launch CUTLASS GEMM via dispatch
  result = run_cutlass_dispatch(
      tbA, tbB, tbC, wpA, wpB, wpC, instA, instB, instC,
      args.M, args.N, args.K,
      args.alpha,
      dA, lda, dB, ldb, dC, ldc,
      args.beta,
      args.iters
  );

  CUDA_CHECK_MSG(result, "Gemm Kernel Launch: ")

  free(hA); free(hB); free(hC);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);

  return 0;

}







