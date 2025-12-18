#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm> // For std::max

using namespace nvcuda;

// ==============================================================================================
// 1. CONFIGURATION & CONSTANTS
// ==============================================================================================

// --- Tensor Core Dimensions (Hardware Fixed) ---
#define M 16
#define N 16
#define K 16

// --- Memory Tiling Configuration (Tunable) ---
// These define the size of the contiguous blocks in Global Memory.
// They must be multiples of 16.
#define MEM_TILE_M 16
#define MEM_TILE_K 128
#define MEM_TILE_N 16

// --- GEMM Problem Size ---
// Total Dimensions must be multiples of the Block size AND Memory Tile size
#define M_TILES 256   // Total rows = 64 * 16 = 1024
#define N_TILES 256   // Total cols = 64 * 16 = 1024
#define K_TILES 8   // Total K    = 64 * 16 = 1024

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

// --- Kernel Implementation Constants ---
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

// Shared Memory Chunking
#define CHUNK_K 8 
#define SKEW_HALF 16 // Avoid bank conflicts

// Layout Constants
#define C_LAYOUT wmma::mem_row_major
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

// Strides
#define GLOBAL_MEM_STRIDE N_GLOBAL
#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// Copy Constants
#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

// ==============================================================================================
// 2. INDEXING MACROS (Must match Host Tiling Logic)
// ==============================================================================================

// Macro to find address in Row-Major Tiled Matrix A
#define GET_A_INDEX_GENERAL(row, col) \
    ( ( ((row)/MEM_TILE_M) * (K_GLOBAL/MEM_TILE_K) + ((col)/MEM_TILE_K) ) * (MEM_TILE_M * MEM_TILE_K) + \
      ( ((row)%MEM_TILE_M) * MEM_TILE_K + ((col)%MEM_TILE_K) ) )

// Macro to find address in Column-Major Tiled Matrix B
#define GET_B_INDEX_GENERAL(row_k, col_n) \
    ( ( ((col_n)/MEM_TILE_N) * (K_GLOBAL/MEM_TILE_K) + ((row_k)/MEM_TILE_K) ) * (MEM_TILE_K * MEM_TILE_N) + \
      ( ((col_n)%MEM_TILE_N) * MEM_TILE_K + ((row_k)%MEM_TILE_K) ) )

// Error handling wrapper
#define checkCudaErrors(func) { \
    cudaError_t e = (func); \
    if (e != cudaSuccess) { \
        printf("CUDA Error %d: %s\n", e, cudaGetErrorString(e)); \
        exit(-1); \
    } \
}

// --- Benchmarking Settings ---
#define WARMUP_ITERS 5
#define BENCH_ITERS 50

// ==============================================================================================
// 3. HOST HELPER FUNCTIONS
// ==============================================================================================

// Initialize matrices with random values
void init_matrix(half *arr, int size) {
    for (int i = 0; i < size; i++) {
        float val = static_cast<float>(rand() % 5) - 2.0f; // Small integers [-2, 2]
        arr[i] = __float2half(val);
    }
}

// Host: Convert Linear A -> Tiled A
void convert_A_to_tiled(const half* src, half* dst) {
    int tiles_per_row = K_GLOBAL / MEM_TILE_K;
    int elements_per_tile = MEM_TILE_M * MEM_TILE_K;

    for (int row = 0; row < M_GLOBAL; row++) {
        for (int col = 0; col < K_GLOBAL; col++) {
            int src_idx = row * K_GLOBAL + col;
            
            int tile_r = row / MEM_TILE_M;
            int tile_c = col / MEM_TILE_K;
            int local_r = row % MEM_TILE_M;
            int local_c = col % MEM_TILE_K;

            int dst_idx = (tile_r * tiles_per_row + tile_c) * elements_per_tile + 
                          (local_r * MEM_TILE_K + local_c);
            
            dst[dst_idx] = src[src_idx];
        }
    }
}

// Host: Convert Linear B -> Tiled B
void convert_B_to_tiled(const half* src, half* dst) {
    int tiles_per_col = K_GLOBAL / MEM_TILE_K; 
    int elements_per_tile = MEM_TILE_K * MEM_TILE_N;

    for (int k = 0; k < K_GLOBAL; k++) {
        for (int n = 0; n < N_GLOBAL; n++) {
            // Src is Column Major: src[n * K + k]
            int src_idx = n * K_GLOBAL + k;

            int tile_k = k / MEM_TILE_K;
            int tile_n = n / MEM_TILE_N;
            int local_k = k % MEM_TILE_K;
            int local_n = n % MEM_TILE_N;

            int dst_idx = (tile_n * tiles_per_col + tile_k) * elements_per_tile + 
                          (local_n * MEM_TILE_K + local_k); // Col Major inside tile

            dst[dst_idx] = src[src_idx];
        }
    }
}

// CPU Reference GEMM (Naive implementation for correctness check)
// C = A * B
void cpu_gemm(const half *A, const half *B, float *C) {
    printf("Computing CPU Reference (this might take a moment)...\n");
    for (int i = 0; i < M_GLOBAL; i++) {
        for (int j = 0; j < N_GLOBAL; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K_GLOBAL; k++) {
                // A is Row Major: A[i][k]
                float a = __half2float(A[i * K_GLOBAL + k]);
                // B is Column Major: B[j][k] (logically B[k][j])
                // The input B array is stored as flat columns.
                // Element at Row k, Col j is at index j * K + k
                float b = __half2float(B[j * K_GLOBAL + k]);
                sum += a * b;
            }
            C[i * N_GLOBAL + j] = sum;
        }
    }
}

// ==============================================================================================
// 4. CUDA KERNEL
// ==============================================================================================

__global__ void compute_gemm_general_tiled(const half *A, const half *B, float *C) {
    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    float *shmem_warp_tile_ptr = (float *)&shmem[0][0] + (warpId / 2) * SHMEM_STRIDE * K * 2 + (warpId % 2) * SHMEM_OFFSET;
    float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_ROW_TILES) % N_TILES;

        if (block_tile_i >= M_TILES) break;

        // Init Accumulators
        wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                wmma::fill_fragment(c[i][j], 0.0f);
            }
        }

        // K-Loop
        #pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            
            // --- Copy Phase (Global Tiled -> Shared Linear) ---
            size_t shmem_idx = warpId < (WARPS_PER_BLOCK / 2)
                                 ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                                 : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

            #pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                int logical_row, logical_col;
                const half* src_matrix;
                size_t src_offset;

                if (warpId < 4) { // Matrix A
                    src_matrix = A;
                    logical_row = (block_tile_i * M) 
                                + (warpId % 4) * (M * 2) 
                                + (laneId / CHUNK_COPY_LINE_LANES) 
                                + (i * CHUNK_COPY_LINES_PER_WARP);
                    logical_col = (tile_k * K) + (laneId % CHUNK_COPY_LINE_LANES) * 8;
                    src_offset = GET_A_INDEX_GENERAL(logical_row, logical_col);
                } else { // Matrix B
                    src_matrix = B;
                    logical_row = (tile_k * K) + (laneId % CHUNK_COPY_LINE_LANES) * 8;
                    logical_col = (block_tile_j * N) 
                                + (warpId % 4) * (N * 2) 
                                + (laneId / CHUNK_COPY_LINE_LANES) 
                                + (i * CHUNK_COPY_LINES_PER_WARP);
                    src_offset = GET_B_INDEX_GENERAL(logical_row, logical_col);
                }

                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = 
                    *((int4 *)(src_matrix + src_offset));

                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }
            __syncthreads();

            // --- Compute Phase ---
            #pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

                #pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t      shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const half *tile_ptr    = &shmem[shmem_idx_a][k_step * K];
                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

                    #pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            size_t      shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
                            const half *tile_ptr    = &shmem[shmem_idx_b][k_step * K];
                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
                        }
                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            __syncthreads();
        }

        // --- Store Phase ---
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }
        __syncthreads();

        const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        float *dst_gmem_warp_stream_ptr = &C[gmem_idx];
        
        #pragma unroll
        for (int i = 0; i < K; i++) {
            *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) = 
                *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }
        __syncthreads();
    }
}

// ==============================================================================================
// 5. MAIN
// ==============================================================================================

int main() {
    printf("Matrix Size: %dx%d (K=%d)\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    printf("Memory Tile Size: %dx%d (M), %dx%d (K), %dx%d (N)\n", 
            MEM_TILE_M, MEM_TILE_K, MEM_TILE_K, MEM_TILE_M, MEM_TILE_N, MEM_TILE_K);

    size_t size_A = M_GLOBAL * K_GLOBAL * sizeof(half);
    size_t size_B = K_GLOBAL * N_GLOBAL * sizeof(half);
    size_t size_C = M_GLOBAL * N_GLOBAL * sizeof(float);

    // 1. Allocate Host Memory
    half *h_A = (half *)malloc(size_A);
    half *h_B = (half *)malloc(size_B);
    half *h_A_tiled = (half *)malloc(size_A);
    half *h_B_tiled = (half *)malloc(size_B);
    float *h_C_gpu = (float *)malloc(size_C);
    float *h_C_ref = (float *)malloc(size_C);

    // 2. Initialize Data
    srand(42);
    init_matrix(h_A, M_GLOBAL * K_GLOBAL);
    init_matrix(h_B, K_GLOBAL * N_GLOBAL);

    // 3. Convert to Tiled Layout on Host
    printf("Converting matrices to Tiled Layout...\n");
    convert_A_to_tiled(h_A, h_A_tiled);
    convert_B_to_tiled(h_B, h_B_tiled);

    // 4. Allocate Device Memory
    half *d_A, *d_B;
    float *d_C;
    checkCudaErrors(cudaMalloc(&d_A, size_A));
    checkCudaErrors(cudaMalloc(&d_B, size_B));
    checkCudaErrors(cudaMalloc(&d_C, size_C));

    // 5. Copy Data to GPU
    checkCudaErrors(cudaMemcpy(d_A, h_A_tiled, size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B_tiled, size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_C, 0, size_C)); // Clear C

    // 6. Launch Kernel
    // Shared mem size needs to be calculated manually
    size_t shmem_size = sizeof(half) * (CHUNK_K * K + SKEW_HALF) * (BLOCK_COL_TILES * M + BLOCK_ROW_TILES * N);
    // Actually we declared `extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF]`
    // The height of shmem is needed. From code:
    // A part uses: BLOCK_COL_TILES * M rows
    // B part uses: BLOCK_ROW_TILES * N rows
    // Total rows: (BLOCK_COL_TILES * M) + (BLOCK_ROW_TILES * N)
    size_t shmem_rows = (BLOCK_COL_TILES * M) + (BLOCK_ROW_TILES * N);
    size_t shmem_bytes = shmem_rows * (CHUNK_K * K + SKEW_HALF) * sizeof(half);

    printf("Shared Memory Required: %zu KB\n", shmem_bytes / 1024);
    
    // Set Shared Mem Attribute if needed (for >48KB)
    cudaFuncSetAttribute(compute_gemm_general_tiled, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

    // dim3 gridDim(M_GLOBAL / (BLOCK_ROW_TILES * M) * N_GLOBAL / (BLOCK_COL_TILES * N)); 
    // Simplified Grid Calculation based on sample code logic:
    // The sample uses a 1D grid launch that iterates over the 2D problem.
    // 1. Create a device property struct
    cudaDeviceProp prop;
    
    // 2. Get the properties for the current device (device 0)
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    
    // 3. Now you can access the member variable from the instance 'prop'
    // int num_blocks = num_blocks * 2;
    
    // Let's use the sample's loop strategy: Grid can be anything, threads loop.
    // We'll set grid size to cover the matrix for simplicity without loop overlap if possible,
    // or just large enough.
    // Block computes (BLOCK_ROW_TILES * M) x (BLOCK_COL_TILES * N) = 128 x 128 elements.
    // Matrix is 1024x1024. Need (1024/128) * (1024/128) = 8 * 8 = 64 blocks.
    // gridDim = dim3(64, 1, 1);
    // dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
    int num_blocks = 2*prop.multiProcessorCount;
    printf("Launching Kernel...\n");
    compute_gemm_general_tiled<<<num_blocks, THREADS_PER_BLOCK, shmem_bytes>>>(d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 7. Copy Result Back
    checkCudaErrors(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));

    // // 8. Verify Correctness
    // cpu_gemm(h_A, h_B, h_C_ref);

    // printf("Verifying Results...\n");
    // float max_diff = 0.0f;
    // float avg_diff = 0.0f;
    // for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
    //     float diff = fabs(h_C_gpu[i] - h_C_ref[i]);
    //     if (diff > max_diff) max_diff = diff;
    //     avg_diff += diff;
    // }
    // avg_diff /= (M_GLOBAL * N_GLOBAL);

    // printf("Max Absolute Error: %f\n", max_diff);
    // printf("Avg Absolute Error: %f\n", avg_diff);

    // if (avg_diff < 0.1f) {
    //     printf("TEST PASSED!\n");
    // } else {
    //     printf("TEST FAILED!\n");
    // }

    // 2. Warm-Up
    printf("Warming up (%d iters)...\n", WARMUP_ITERS);
    for (int i = 0; i < WARMUP_ITERS; i++) {
        compute_gemm_general_tiled<<<num_blocks, THREADS_PER_BLOCK, shmem_bytes>>>(d_A, d_B, d_C);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    // 3. Timing Loop
    printf("Benchmarking (%d iters)...\n", BENCH_ITERS);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        compute_gemm_general_tiled<<<num_blocks, THREADS_PER_BLOCK, shmem_bytes>>>(d_A, d_B, d_C);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float total_msec = 0;
    checkCudaErrors(cudaEventElapsedTime(&total_msec, start, stop));

    // ==============================================================================================
    // TFLOPS CALCULATION
    // ==============================================================================================
    
    // Total Operations: 2 * M * N * K (multiply-add counts as 2 ops)
    double ops = 2.0 * (double)M_GLOBAL * (double)N_GLOBAL * (double)K_GLOBAL;
    double avg_msec = total_msec / BENCH_ITERS;
    double avg_sec = avg_msec / 1000.0;
    double tflops = (ops / avg_sec) / 1e12;

    printf("\n----------------------------------------------------------------\n");
    printf("Performance Results:\n");
    printf("  Avg Time: %.4f msec\n", avg_msec);
    printf("  Throughput: %.2f TFLOPS\n", tflops);
    printf("----------------------------------------------------------------\n");

    // Cleanup
    free(h_A); free(h_B); free(h_A_tiled); free(h_B_tiled); free(h_C_gpu); free(h_C_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}