#include "kernels.hpp"

/**
 * The most basic version of a GEMM kernel.
 */

__global__ void kernel_v00_basic(KernelArgs args) {
    UNPACK_KERNEL_ARGS(args);

    printf("Thread %d\n", threadIdx.x);
}