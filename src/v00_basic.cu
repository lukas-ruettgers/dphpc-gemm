#include "kernels.hpp"

/**
 * The most basic version of a GEMM kernel.
 */

__global__ void kernel_v00_basic(KernelArgs args) {
    UNPACK_KERNEL_ARGS(args);

    // Map block and thread IDs to row and column.
    const int row_C = blockIdx.x * blockDim.x + threadIdx.x;
    const int col_C = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check for non-multiple of block size.
    if (row_C >= M || col_C >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k_idx = 0; k_idx < K; ++k_idx) {
        acc += A[row_C * K + k_idx] * B[k_idx * N + col_C];
    }
    
    const int output_idx = row_C * N + col_C;
    C[output_idx] += acc;
}