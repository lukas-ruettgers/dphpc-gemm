#include "utils.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <iostream>

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


int main() {
    int M = 8192, N = 8192, K = 512;

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t; 
    using ElementC = float;
    using ElementAccumulator = float;

    // For small K, consider RowMajor for both A and B
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;  // Changed from ColumnMajor
    using LayoutC = cutlass::layout::RowMajor;

   
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB, 
        ElementC, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 32>,  // Threadblock tile
        cutlass::gemm::GemmShape<64, 64, 32>,    // Warp tile  
        cutlass::gemm::GemmShape<16, 8, 8>,      // Instruction tile
        cutlass::epilogue::thread::LinearCombination<
            ElementC,
            1,
            ElementAccumulator,
            ElementAccumulator
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;  // Stages

    // Initialize Matrix

    size_t size_A = size_t(M) * size_t(K);
    size_t size_B = size_t(K) * size_t(N);
    size_t size_C = size_t(M) * size_t(N);

    // host buffers (fp32 host init -> convert)
    ElementA *hA = alloc_host<ElementA>(size_A);
    ElementB *hB = alloc_host<ElementB>(size_B);
    ElementC *hC = alloc_host<ElementC>(size_C);

    fill_random<ElementA>(hA, size_A);
    fill_random<ElementB>(hB, size_B);
    for (size_t i=0;i<size_C;i++) hC[i] = cutlass::half_t(0.0f);

    ElementA *dA = alloc_device<ElementA>(size_A);
    ElementB *dB = alloc_device<ElementB>(size_B);
    ElementC *dC = alloc_device<ElementC>(size_C);

    checkCuda(cudaMemcpy(dA, hA, size_A*sizeof(ElementA), cudaMemcpyHostToDevice), "memcpy A");
    checkCuda(cudaMemcpy(dB, hB, size_B*sizeof(ElementB), cudaMemcpyHostToDevice), "memcpy B");
    checkCuda(cudaMemcpy(dC, hC, size_C*sizeof(ElementC), cudaMemcpyHostToDevice), "memcpy C");

    Gemm gemm_op;
    
    // Create arguments - note the leading dimension changes due to layout
    typename Gemm::Arguments args(
        {M, N, K},  // Problem size
        {dA, K},     // A: RowMajor, lda = K  
        {dB, N},     // B: RowMajor, ldb = N (was K with ColumnMajor)
        {dC, N},     // C: RowMajor, ldc = N
        {dC, N},     // D: RowMajor, ldd = N  
        {1.0f, 0.0f}  // Epilogue values
    );

    // Check if initialization succeeded
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to initialize CUTLASS gemm: " 
                  << cutlass::cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    // Run and time the operation
    double t_ms = time_cuda_event([&]() { 
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Failed to run CUTLASS gemm: "
                      << cutlass::cutlassGetStatusString(status) << std::endl;
        }
    });

    if (status == cutlass::Status::kSuccess) {
        double tflops = 2.0 * M * N * K / (t_ms * 1e9);
        std::cout << "CUTLASS: M=" << M << " N=" << N << " K=" << K
                  << "  Time=" << t_ms << " ms  Perf=" << tflops << " TFLOPs\n";
    }

    // Cleanup
    cudaFree(dA); 
    cudaFree(dB); 
    cudaFree(dC); 

    free(hA); free(hB); free(hC);
    
    return 0;
}