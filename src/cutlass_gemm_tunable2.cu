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
  int INST_M, int INST_N, int INST_K,
  int STAGES
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
  int iters,
  double &t_ms){
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
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, STAGES>; // default swizzle


    typename Gemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                        {A, lda},    // Tensor-ref for source matrix A
                        {B, ldb},    // Tensor-ref for source matrix B
                        {C, ldc},    // Tensor-ref for source matrix C
                        {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                        {alpha, beta}); // Scalars used in the Epilogue

    Gemm gemm_op;
    gemm_op.initialize(args);
    // CUDA_CHECK_MSG_RET(gemm_op.initialize(args), "Gemm initialize failed: ")


    t_ms = time_cuda_event([&]() { gemm_op(); }, 3, iters); //handles warmup

    return cudaSuccess;
}

template<int BM, int BN, int BK, int WM, int WN, int WK, int IM, int IN, int IK, int STAGES>
cudaError_t launch_gemm(
    int M, int N, int K,
    float alpha,
    const cutlass::half_t *A, int lda,
    const cutlass::half_t *B, int ldb,
    float beta,
    float *C, int ldc,
    int iters, double &t_ms)
{
    return CutlassMgemmNN<BM, BN, BK, WM, WN, WK, IM, IN, IK, STAGES>(
        M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, iters, t_ms
    );
}

// -----------------------------------------------------------------------------
// Dispatch table: map runtime tile choice -> precompiled template call
// -----------------------------------------------------------------------------
#define CFGCOUNT 26  //update this when you add a new config

struct GemmConfigEntry
{
  int BM, BN, BK;
  int WM, WN, WK;
  int IM, IN, IK;
  int stages = 2;
};

void printConfig(GemmConfigEntry cfg){
  std::cout << "Config: TB=" << cfg.BM << "x" << cfg.BN << "x" << cfg.BK
            << " WP=" << cfg.WM << "x" << cfg.WN << "x" << cfg.WK
            << " INST=" << cfg.IM << "x" << cfg.IN << "x" << cfg.IK << " Stages="<<cfg.stages; 
}

constexpr GemmConfigEntry kConfigs[] = {
  {128,256,32,64,64,32,8,8,4,2},
  {128,256,32,64,64,32,16,8,8,2},
  {128,128,32,64,64,32,8,8,4,2},
  {128,128,32,64,64,32,16,8,8,2},
  {256,128,32,64,64,32,8,8,4,2},
  {256,128,32,64,64,32,16,8,8,2},
  {128, 256, 64, 64, 64, 64, 16, 8, 16, 3},
  {64, 256, 32, 32, 64, 32, 16, 8, 16, 4},
  {128, 128, 32, 64, 64, 32, 16, 8, 16, 4},
  {128, 64, 32, 64, 32, 32, 16, 8, 16, 4},
  {64, 128, 32, 32, 64, 32, 16, 8, 16, 4},
  {128, 32, 32, 64, 32, 32, 16, 8, 16, 4},
  {64, 32, 32, 32, 32, 32, 16, 8, 16, 5},
  {32, 64, 32, 32, 32, 32, 16, 8, 16, 5},
  {128, 128, 64, 64, 64, 64, 16, 8, 16, 4},
  {128, 64, 64, 64, 32, 64, 16, 8, 16, 4},
  {64, 128, 64, 32, 64, 64, 16, 8, 16, 4},
  {256, 256, 32, 64, 64, 32, 16, 8, 16, 3},
  {256, 128, 32, 64, 64, 32, 16, 8, 16, 3},
  {128, 256, 32, 64, 64, 32, 16, 8, 16, 3},
  {64, 64, 32, 32, 32, 32, 16, 8, 16, 5},
  {256, 256, 64, 64, 64, 64, 16, 8, 16, 3},
  {256, 128, 64, 64, 64, 64, 16, 8, 16, 3},
  {128, 256, 64, 64, 64, 64, 16, 8, 16, 4},
  {256, 256, 64, 64, 64, 64, 16, 8, 16, 4},
  {128, 128, 64, 64, 64, 64, 16, 8, 16, 3},
};

int get_config_idx(GemmConfigEntry Gemmcfg){
  for(int i = 0;i<CFGCOUNT;i++){
    if(kConfigs[i].BM == Gemmcfg.BM && kConfigs[i].BN == Gemmcfg.BN && kConfigs[i].BK == Gemmcfg.BK &&
       kConfigs[i].WM == Gemmcfg.WM && kConfigs[i].WN == Gemmcfg.WN && kConfigs[i].WK == Gemmcfg.WK &&
       kConfigs[i].IM == Gemmcfg.IM && kConfigs[i].IN == Gemmcfg.IN && kConfigs[i].IK == Gemmcfg.IK){
        return i;
       }
  }
  return -1;
}

using GemmFn = cudaError_t (*)(
    int M, int N, int K,
    float alpha,
    const cutlass::half_t *A, int lda,
    const cutlass::half_t *B, int ldb,
    float beta,
    float *C, int ldc,
    int iters, double &t_ms
);

constexpr GemmFn kernel_table[] = {
    launch_gemm<128,256,32, 64,64,32, 8,8,4,2>,
    launch_gemm<128,256,32, 64,64,32, 16,8,8,2>,
    launch_gemm<128,128,32, 64,64,32, 8,8,4,2>,
    launch_gemm<128,128,32, 64,64,32, 16,8,8,2>,
    launch_gemm<256,128,32, 64,64,32, 8,8,4,2>,
    launch_gemm<256,128,32, 64,64,32, 16,8,8,2>,
    launch_gemm<128, 256, 64, 64, 64, 64, 16,8,8, 3>,
    launch_gemm<64, 256, 32, 32, 64, 32, 16, 8, 16, 4>,
    launch_gemm<128, 128, 32, 64, 64, 32, 16, 8, 16, 4>,
    launch_gemm<128, 64, 32, 64, 32, 32, 16, 8, 16, 4>,
    launch_gemm<64, 128, 32, 32, 64, 32, 16, 8, 16, 4>,
    launch_gemm<128, 32, 32, 64, 32, 32, 16, 8, 16, 4>,
    launch_gemm<64, 32, 32, 32, 32, 32, 16, 8, 16, 5>,
    launch_gemm<32, 64, 32, 32, 32, 32, 16, 8, 16, 5>,
    launch_gemm<128, 128, 64, 64, 64, 64, 16, 8, 16, 4>,
    launch_gemm<128, 64, 64, 64, 32, 64, 16, 8, 16, 4>,
    launch_gemm<64, 128, 64, 32, 64, 64, 16, 8, 16, 4>,
    launch_gemm<256, 256, 32, 64, 64, 32, 16, 8, 16, 3>,
    launch_gemm<256, 128, 32, 64, 64, 32, 16, 8, 16, 3>,
    launch_gemm<128, 256, 32, 64, 64, 32, 16, 8, 16, 3>,
    launch_gemm<64, 64, 32, 32, 32, 32, 16, 8, 16, 5>,
    launch_gemm<256, 256, 64, 64, 64, 64, 16, 8, 16, 3>,
    launch_gemm<256, 128, 64, 64, 64, 64, 16, 8, 16, 3>,
    launch_gemm<128, 256, 64, 64, 64, 64, 16, 8, 16, 4>,
    launch_gemm<256, 256, 64, 64, 64, 64, 16, 8, 16, 4>,
    launch_gemm<128, 128, 64, 64, 64, 64, 16, 8, 16, 3>,
};

cudaError_t run_cutlass_dispatch(
    GemmConfigEntry Gemmcfg,
    int M, int N, int K,
    float alpha,
    cutlass::half_t const *dA, int lda,
    cutlass::half_t const *dB, int ldb,
    float *dC, int ldc,
    float beta,
    int iters, double &t_ms)
{
 
  int config_id = get_config_idx(Gemmcfg);
  
  if(config_id != -1){
    return kernel_table[config_id](M, N, K, alpha, dA, lda, dB, ldb, beta, dC, ldc, iters, t_ms);
  }
  
  // fallback: unsupported
  std::cerr << "run_cutlass_dispatch: unsupported tile/warp configuration";
  //           << tbA << "x" << tbB << "x" << tbC << " TB, "
  //           << wpA << "x" << wpB << "x" << wpC << " WP\n"
  //           << instA << "x" << instB << "x" << instC << " WP\n";
  // std::cerr << "Add a matching CutlassMgemmNN<...> instantiation to the binary.\n";
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
  bool autotune = false;
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
    else if (a=="--autotune") args.autotune = true;
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

  GemmConfigEntry Gemmcfg;

  parse_shape(args.threadblock, Gemmcfg.BM, Gemmcfg.BN, Gemmcfg.BK);
  parse_shape(args.warp, Gemmcfg.WM, Gemmcfg.WN, Gemmcfg.WK);
  parse_shape(args.inst, Gemmcfg.IM, Gemmcfg.IN, Gemmcfg.IK);
  
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
  double t_ms;
  double flops = 2.0 * args.M * args.N * args.K; 
  if(!args.autotune){
    // Launch CUTLASS GEMM via dispatch
    result = run_cutlass_dispatch(
        Gemmcfg,
        args.M, args.N, args.K,
        args.alpha,
        dA, lda, dB, ldb, dC, ldc,
        args.beta,
        args.iters, t_ms
    );
    double tflops = flops/(t_ms * 1e9);
    printConfig(Gemmcfg);
    std::cout << " CUTLASS: M=" << args.M << " N=" << args.N << " K=" << args.K
              << "  Time=" << t_ms << " ms  Perf=" << tflops << " TFLOPs\n";
    
    CUDA_CHECK_MSG(result, "Gemm Kernel Launch: ")
  }
  else{
    std::cout << " CUTLASS: M=" << args.M << " N=" << args.N << " K=" << args.K <<"\n";
    double max_flops = 0;
    int best_config;
    std::cout << "Searching for best config..." <<"\n";
    for (int i = 0;i< CFGCOUNT; i++){
      result = run_cutlass_dispatch(
        kConfigs[i],
        args.M, args.N, args.K,
        args.alpha,
        dA, lda, dB, ldb, dC, ldc,
        args.beta,
        args.iters, t_ms
      );
      double tflops = flops/(t_ms * 1e9);
      if(tflops>max_flops){
        max_flops = tflops;
        best_config = i;
      }
      printConfig(kConfigs[i]);
      std::cout<<" "<<tflops<<" TFLOPs\n";
    }
    std::cout<<"######## Best Config ############\n";
    printConfig(kConfigs[best_config]);
    std::cout<< "  Perf=" << max_flops << " TFLOPs\n";
  }

  free(hA); free(hB); free(hC);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);

  return 0;

}







