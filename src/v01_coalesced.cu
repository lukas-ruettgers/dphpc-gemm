#include "kernels.hpp"

/**
 * The basic kernel with memory coalescing.
 */

__global__ void kernel_v01_coalesced(KernelContext ctx) {
    UNPACK_KERNEL_ARGS(ctx);

    // Map block and thread IDs to row and column.
    const int row_C = blockIdx.y * blockDim.y + threadIdx.y;
    const int col_C = blockIdx.x * blockDim.x + threadIdx.x;

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


class Kernel_V01_Coalesced : public Kernel {
    public:
    void launch(KernelContext ctx) override {
        UNPACK_KERNEL_ARGS(ctx);

        dim3 block(TB_M, TB_N); // 2D block of TB_M x TB_N threads
        dim3 grid(blocks_M, blocks_N); // 2D grid of blocks_M x blocks_N blocks

        kernel_v01_coalesced<<<grid, block>>>(ctx);
    }
};

DEFINE_KERNEL_INSTANCE(Kernel_V01_Coalesced);