#include "utils.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

int main() {
    int M = 8192, N = 8192, K = 128;  // K small, M,N large

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t, cutlass::layout::RowMajor, //A
        cutlass::half_t, cutlass::layout::ColumnMajor, //B
        float, cutlass::layout::RowMajor>; //C

    cutlass::half_t *A, *B;
    float *C, *D;
    cudaMalloc(&A, sizeof(cutlass::half_t) * M * K);
    cudaMalloc(&B, sizeof(cutlass::half_t) * K * N);
    cudaMalloc(&C, sizeof(float) * M * N);
    cudaMalloc(&D, sizeof(float) * M * N);

    Gemm gemm_op;
    typename Gemm::Arguments args({M, N, K}, {A, K}, {B, K}, {C, N}, {D, N}, {1.0f, 0.0f});

    double t_ms = time_cuda_event([&]() { gemm_op(args); });
    double tflops = 2.0 * M * N * K / (t_ms * 1e9);

    std::cout << "CUTLASS: M=" << M << " N=" << N << " K=" << K
              << "  Time=" << t_ms << " ms  Perf=" << tflops << " TFLOPs\n";

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D);
    return 0;
}
