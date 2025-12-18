#include "utils.hpp"
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

__global__ void wmma_gemm_kernel(half *A, half *B, float *C, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    if (warpM * 16 >= M || warpN * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + (warpM * 16) * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + (warpN * 16), N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + (warpM * 16) * N + (warpN * 16), c_frag, N, wmma::mem_row_major);
}

int main() {
    int M = 8192, N = 8192, K = 128;

    half *A, *B;
    float *C;
    cudaMalloc(&A, sizeof(half) * M * K);
    cudaMalloc(&B, sizeof(half) * K * N);
    cudaMalloc(&C, sizeof(float) * M * N);

    dim3 block(128, 4);
    dim3 grid((M + 15)/16, (N + 15)/16);
    auto func = [&]() {
        wmma_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
        cudaDeviceSynchronize();
    };

    double t_ms = time_cuda_event(func);
    double tflops = 2.0 * M * N * K / (t_ms * 1e9);

    std::cout << "WMMA: M=" << M << " N=" << N << " K=" << K
              << "  Time=" << t_ms << " ms  Perf=" << tflops << " TFLOPs\n";

    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
