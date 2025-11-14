// cutlass_gemm_tunable.cu
//
// Single-file example that exposes many CUTLASS hyper-parameters via CLI.
// Designed for small K, large M/N workloads (e.g., M=N=4096, K=4..16).
//
// Build (example):
// nvcc -I/path/to/cutlass/include -I/path/to/cutlass/tools -O3 \
//   -std=c++17 cutlass_gemm_tunable.cu -o cutlass_gemm_tunable \
//   -lcudart -lcublas \
//   -gencode=arch=compute_80,code=sm_80   # replace with target GPU SM
//
// Run example:
// ./cutlass_gemm_tunable --M 4096 --N 4096 --K 8 --dtype fp16 \
//    --threadblock 128x128x8 --warp 64x64x8 --inst 16x8x8 \
//    --splitk 1 --batch 1 --alpha 1 --beta 0 --iters 20
//
// NOTES:
//  - If the exact threadblock/warp/inst shapes are not compiled-in this binary,
//    it will fall back to a default instantiation. Add more instantiations
//    to the 'kernel_registry' if you need other compile-time shapes.
//
//

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

#include <cuda_runtime.h>

// Simple CLI parsing helper (no dependency).
struct Args {
  int M = 4096, N = 4096, K = 8;
  int batch = 1;
  int split_k_slices = 1;
  std::string dtype = "fp16"; // fp16 or fp32
  std::string threadblock = "128x128x8";
  std::string warp = "64x64x8";
  std::string inst = "16x8x8";
  int iters = 20;
  float alpha = 1.0f, beta = 0.0f;
  bool use_tensorop = true;
};

static void checkCuda(cudaError_t res, const char* msg = "") {
  if (res != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(res) << std::endl;
    std::exit(EXIT_FAILURE);
  }
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

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i=1;i<argc;i++){
    std::string a(argv[i]);
    if (a=="--M") args.M = atoi(argv[++i]);
    else if (a=="--N") args.N = atoi(argv[++i]);
    else if (a=="--K") args.K = atoi(argv[++i]);
    else if (a=="--batch") args.batch = atoi(argv[++i]);
    else if (a=="--splitk") args.split_k_slices = atoi(argv[++i]);
    else if (a=="--dtype") args.dtype = argv[++i];
    else if (a=="--threadblock") args.threadblock = argv[++i];
    else if (a=="--warp") args.warp = argv[++i];
    else if (a=="--inst") args.inst = argv[++i];
    else if (a=="--iters") args.iters = atoi(argv[++i]);
    else if (a=="--alpha") args.alpha = atof(argv[++i]);
    else if (a=="--beta") args.beta = atof(argv[++i]);
    else if (a=="--no-tensorop") { args.use_tensorop = false; }
    else {
      std::cerr << "Unknown arg: " << a << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  return args;
}

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

//
// ------------------ Kernel instantiations / registry ------------------
//
// We instantiate a handful of commonly useful GEMM kernels for the 
// small-K regime. Add more compile-time combos here if you want
// to experiment. Each instantiation differs by Threadblock/Warp/Instr shapes.
//
// The registry maps a textual key -> a callable wrapper that runs the kernel.
//
// For readability we provide two datatypes: fp16 (tensorop) and fp32 (simt).
//

// Utility to produce descriptive key strings
std::string mk_key(const std::string &tb, const std::string &wp, const std::string &inst, const std::string &dtype) {
  return dtype + ":" + tb + ":" + wp + ":" + inst;
}

// We define a base 'KernelRunner' interface
struct KernelRunnerBase {
  virtual ~KernelRunnerBase() = default;
  virtual bool matches(const std::string &dtype) const = 0;
  virtual bool run(int M, int N, int K, int batch, int split_k_slices, float alpha, float beta, int iters) = 0;
};

using KernelRunnerPtr = std::shared_ptr<KernelRunnerBase>;

//
// Example instantiation 1: fp16, TensorOp, Threadblock 128x128x8, Warp 64x64x8, Inst 16x8x8
//
template <
  int TB_M, int TB_N, int TB_K,
  int WP_M, int WP_N, int WP_K,
  int INST_M, int INST_N, int INST_K
>
struct GemmRunner_fp16_tb128_64_16 : public KernelRunnerBase {

  bool matches(const std::string &dtype) const override {
    return dtype == "fp16";
  }

  bool run(int M, int N, int K, int batch, int split_k_slices, float alpha, float beta, int iters) override {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, // keep fairly generic; compile with proper -gencode
        cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,
        cutlass::gemm::GemmShape<WP_M, WP_N, WP_K>,
        cutlass::gemm::GemmShape<INST_M, INST_N, INST_K>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC,
            1,
            ElementAccumulator,
            ElementAccumulator
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // default swizzle
        2 // stages (pipeline depth)
    >;

    // shapes
    int64_t strideA = K;
    int64_t strideB = N;
    int64_t strideC = N;

    size_t size_A = size_t(M) * size_t(K) * batch;
    size_t size_B = size_t(K) * size_t(N) * batch;
    size_t size_C = size_t(M) * size_t(N) * batch;

    // host buffers (fp32 host init -> convert)
    ElementA *hA = alloc_host<ElementA>(size_A);
    ElementB *hB = alloc_host<ElementB>(size_B);
    ElementC *hC = alloc_host<ElementC>(size_C);
    for (size_t i=0;i<size_A;i++) hA[i] = cutlass::half_t(((float)rand() / RAND_MAX - 0.5f));
    for (size_t i=0;i<size_B;i++) hB[i] = cutlass::half_t(((float)rand() / RAND_MAX - 0.5f));
    for (size_t i=0;i<size_C;i++) hC[i] = cutlass::half_t(0.0f);

    ElementA *dA = alloc_device<ElementA>(size_A);
    ElementB *dB = alloc_device<ElementB>(size_B);
    ElementC *dC = alloc_device<ElementC>(size_C);

    checkCuda(cudaMemcpy(dA, hA, size_A*sizeof(ElementA), cudaMemcpyHostToDevice), "memcpy A");
    checkCuda(cudaMemcpy(dB, hB, size_B*sizeof(ElementB), cudaMemcpyHostToDevice), "memcpy B");
    checkCuda(cudaMemcpy(dC, hC, size_C*sizeof(ElementC), cudaMemcpyHostToDevice), "memcpy C");

    // Construct GEMM params
    typename Gemm::Arguments args(
      { M, N, K },
      { dA, strideA },
      { dB, strideB },
      { dC, strideC },
      { dC, strideC },
      { alpha, beta },
      split_k_slices
    );

    Gemm gemm_op;
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm initialize failed: " << int(status) << std::endl;
      return false;
    }

    // warmup
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm run failed on warmup: " << int(status) << std::endl;
      return false;
    }
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    // timing
    cudaEvent_t ev1, ev2;
    checkCuda(cudaEventCreate(&ev1));
    checkCuda(cudaEventCreate(&ev2));
    checkCuda(cudaEventRecord(ev1));

    for (int i=0;i<iters;i++) {
      status = gemm_op();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Gemm run failed: " << int(status) << std::endl;
        return false;
      }
    }
    checkCuda(cudaEventRecord(ev2));
    checkCuda(cudaEventSynchronize(ev2));

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, ev1, ev2));
    float avg_ms = ms / float(iters);

    double flops = double(2.0) * double(M) * double(N) * double(K) * double(batch) / double(split_k_slices);
    double gflops = (flops / 1e9) / (avg_ms / 1000.0);

    std::cout << "Kernel: fp16 TB=" << TB_M << "x" << TB_N << "x" << TB_K
              << " WP=" << WP_M << "x" << WP_N << "x" << WP_K
              << " INST=" << INST_M << "x" << INST_N << "x" << INST_K
              << " :: M=" << M << " N=" << N << " K=" << K
              << " batch=" << batch << " splitK=" << split_k_slices
              << " avg_ms=" << avg_ms << " gflops=" << gflops << std::endl;

    // cleanup
    checkCuda(cudaFree(dA)); checkCuda(cudaFree(dB)); checkCuda(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return true;
  }
};

//
// Example instantiation 2: fp32 SIMT, Threadblock 128x128x8, Warp 64x64x8, Inst 8x8x4 (instruction shape not used for SIMT but compile requires it)
//
template <
  int TB_M, int TB_N, int TB_K,
  int WP_M, int WP_N, int WP_K,
  int INST_M, int INST_N, int INST_K
>
struct GemmRunner_fp32_tb128_64_8 : public KernelRunnerBase {

  bool matches(const std::string &dtype) const override {
    return dtype == "fp32";
  }

  bool run(int M, int N, int K, int batch, int split_k_slices, float alpha, float beta, int iters) override {
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,
        cutlass::gemm::GemmShape<WP_M, WP_N, WP_K>,
        cutlass::gemm::GemmShape<INST_M, INST_N, INST_K>,
        cutlass::epilogue::thread::LinearCombination<
            ElementC,
            1,
            ElementAccumulator,
            ElementAccumulator
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2
    >;

    // sizes
    int64_t strideA = K;
    int64_t strideB = N;
    int64_t strideC = N;
    size_t size_A = size_t(M) * size_t(K) * batch;
    size_t size_B = size_t(K) * size_t(N) * batch;
    size_t size_C = size_t(M) * size_t(N) * batch;

    float *hA = alloc_host<float>(size_A);
    float *hB = alloc_host<float>(size_B);
    float *hC = alloc_host<float>(size_C);
    for (size_t i=0;i<size_A;i++) hA[i] = (float(rand())/RAND_MAX - 0.5f);
    for (size_t i=0;i<size_B;i++) hB[i] = (float(rand())/RAND_MAX - 0.5f);
    for (size_t i=0;i<size_C;i++) hC[i] = 0.0f;

    float *dA = alloc_device<float>(size_A);
    float *dB = alloc_device<float>(size_B);
    float *dC = alloc_device<float>(size_C);

    checkCuda(cudaMemcpy(dA, hA, size_A*sizeof(float), cudaMemcpyHostToDevice), "memcpy A");
    checkCuda(cudaMemcpy(dB, hB, size_B*sizeof(float), cudaMemcpyHostToDevice), "memcpy B");
    checkCuda(cudaMemcpy(dC, hC, size_C*sizeof(float), cudaMemcpyHostToDevice), "memcpy C");

    typename Gemm::Arguments args(
      { M, N, K },
      { dA, strideA },
      { dB, strideB },
      { dC, strideC },
      { dC, strideC },
      { alpha, beta },
      split_k_slices
    );

    Gemm gemm_op;
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm initialize failed: " << int(status) << std::endl;
      return false;
    }

    status = gemm_op(); if (status != cutlass::Status::kSuccess) { std::cerr << "warmup fail\n"; return false; }
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t ev1, ev2; checkCuda(cudaEventCreate(&ev1)); checkCuda(cudaEventCreate(&ev2));
    checkCuda(cudaEventRecord(ev1));
    for (int i=0;i<iters;i++) {
      status = gemm_op();
      if (status != cutlass::Status::kSuccess) { std::cerr << "run fail\n"; return false; }
    }
    checkCuda(cudaEventRecord(ev2)); checkCuda(cudaEventSynchronize(ev2));
    float ms = 0.0f; checkCuda(cudaEventElapsedTime(&ms, ev1, ev2));
    float avg_ms = ms/float(iters);

    double flops = double(2.0) * double(M) * double(N) * double(K) * double(batch) / double(split_k_slices);
    double gflops = (flops / 1e9) / (avg_ms / 1000.0);

    std::cout << "Kernel: fp32 TB=" << TB_M << "x" << TB_N << "x" << TB_K
              << " WP=" << WP_M << "x" << WP_N << "x" << WP_K
              << " INST=" << INST_M << "x" << INST_N << "x" << INST_K
              << " :: M=" << M << " N=" << N << " K=" << K
              << " avg_ms=" << avg_ms << " gflops=" << gflops << std::endl;

    checkCuda(cudaFree(dA)); checkCuda(cudaFree(dB)); checkCuda(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return true;
  }
};

//
// Build a registry of available kernels. Add more instantiations here.
// The key is: dtype:TB:WP:INST
//
#include <map>

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv);

  int tbA, tbB, tbC, wpA, wpB, wpC, instA, instB, instC;
  parse_shape(args.threadblock, tbA, tbB, tbC);
  parse_shape(args.warp, wpA, wpB, wpC);
  parse_shape(args.inst, instA, instB, instC);

  std::map<std::string, KernelRunnerPtr> registry;

  // Register a few commonly useful instantiations
  registry[ mk_key("128x128x8","64x64x8","16x8x8","fp16") ] = std::make_shared<
    GemmRunner_fp16_tb128_64_16<128,128,8, 64,64,8, 16,8,8> >();

  registry[ mk_key("128x64x8","64x32x8","16x8x8","fp16") ] = std::make_shared<
    GemmRunner_fp16_tb128_64_16<128,64,8, 64,32,8, 16,8,8> >();

  registry[ mk_key("64x128x8","32x64x8","16x8x8","fp16") ] = std::make_shared<
    GemmRunner_fp16_tb128_64_16<64,128,8, 32,64,8, 16,8,8> >();

  // fp32 SIMT examples
  registry[ mk_key("128x128x8","64x64x8","8x8x4","fp32") ] = std::make_shared<
    GemmRunner_fp32_tb128_64_8<128,128,8, 64,64,8, 8,8,4> >();

  registry[ mk_key("128x64x8","64x32x8","8x8x4","fp32") ] = std::make_shared<
    GemmRunner_fp32_tb128_64_8<128,64,8, 64,32,8, 8,8,4> >();

  // Try to find exact match
  std::string key = mk_key(args.threadblock, args.warp, args.inst, args.dtype);
  auto it = registry.find(key);

  if (it == registry.end()) {
    std::cerr << "Requested kernel not found in binary: " << key << std::endl;
    std::cerr << "Available kernels:\n";
    for (auto &kv : registry) std::cerr << "  " << kv.first << "\n";
    std::cerr << "\nEither recompile with the desired compile-time shapes or pick one listed above.\n";
    return 1;
  }

  // Run chosen kernel
  bool ok = it->second->run(args.M, args.N, args.K, args.batch, args.split_k_slices, args.alpha, args.beta, args.iters);
  if (!ok) { std::cerr << "Kernel run failed\n"; return 2; }
  return 0;
}
