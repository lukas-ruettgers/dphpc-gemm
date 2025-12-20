#include "kernels.hpp"

/**
 * Kernel with shared memory.
 */

// NOTE: We assume all TB sizes are same.

__global__ void kernel_v02_shared_mem(KernelContext ctx) {
    UNPACK_KERNEL_ARGS(ctx);

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // const int block_size_row = blockDim.y;
    // const int block_size_col = blockDim.x;

    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;

    int global_row = block_row * TB_M + thread_row;
    int global_col = block_col * TB_N + thread_col;

    extern __shared__ float shared_mem[]; // Size: (TB_M * TB_K + TB_K * TB_N)
    float *tile_a = shared_mem; // Size: TB_M * TB_K
    float *tile_b = shared_mem + (TB_M * TB_K); // Size: TB_K * TB_N

    // Move pointers.
    A += block_row * TB_M * K; // row=block_row, col=0
    B += block_col * TB_N;              // row=0, col=block_col
    C += block_row * TB_M * N + block_col * TB_N;

    float acc = 0.0f;
    // Loop over all tiles along K dimension.
    for (int k_tile_idx = 0; k_tile_idx < K; k_tile_idx += TB_K) {
        // Load tile from matrix A into shared memory with bounds checking
        // thread_col is consecutive for coalesced memory access
        if (global_row < M && (k_tile_idx + thread_col) < K) {
            tile_a[thread_row * TB_K + thread_col] = A[thread_row * K + thread_col];
        } else {
            tile_a[thread_row * TB_K + thread_col] = 0.0f;
        }

        // Load tile from matrix B into shared memory with bounds checking
        // thread_col is consecutive for coalesced memory access
        if ((k_tile_idx + thread_row) < K && global_col < N){
            tile_b[thread_row * TB_N + thread_col] = B[thread_row * N + thread_col];
        } else {
            tile_b[thread_row * TB_N + thread_col] = 0.0f;
        }

        // Block threads until cache is fully populated
        __syncthreads();

        // Advance pointers to next tile.
        A += TB_K;
        B += TB_K * N;

        // Compute partial dot product using shared memory
        for (int dot_idx = 0; dot_idx < TB_K; ++dot_idx) {
            acc += tile_a[thread_row * TB_K + dot_idx] * tile_b[dot_idx * TB_N + thread_col];
        }

        // Sync again to avoid faster threads fetching next block before slower threads finish
        __syncthreads();
    }
    
    // Write result to global memory with bounds checking: C = α*(A@B)+β*C
    if (global_row < M && global_col < N) {
        C[thread_row * N + thread_col] += acc;
    }
}


class Kernel_V02_Shared_Mem : public Kernel {
    public:
    void launch(KernelContext ctx) override {
        UNPACK_KERNEL_ARGS(ctx);

        // x represents columns (N), y represents rows (M)
        dim3 block(TB_N, TB_M); // 2D block of TB_M x TB_N threads
        dim3 grid(blocks_N, blocks_M); // 2D grid of blocks_M x blocks_N blocks
        size_t shared_mem_size = (TB_M * TB_K + TB_K * TB_N) * sizeof(float);

        kernel_v02_shared_mem<<<grid, block, shared_mem_size>>>(ctx);
    }
};

DEFINE_KERNEL_INSTANCE(Kernel_V02_Shared_Mem);