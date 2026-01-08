// gemm_blocked_example.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>
#include <cassert>

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef BENCHMARK
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define BENCHMARK 1
#endif

#if BENCHMARK
float WARMUP=3;
float ITER=50;
#endif

int M = 8192;
int K = 8192;
int N = 8192;

constexpr int TB_M = 128;
constexpr int TB_N = 128;
constexpr int TB_K = 16;
constexpr int WM = 64;
constexpr int WN = 64;
constexpr int WNITER = 4;
constexpr int TM = 8;
constexpr int TN = 4;
constexpr int NUM_THREADS = 128;

// Derived block counts (runtime)
int BM, BK, BN;

// simple CUDA error-check macro
inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// template void sgemm_warptiling<128, 128, 16, 64, 32, 2, 8, 4, 256>(
//     const torch::Tensor &, const torch::Tensor &, torch::Tensor &, float, float);

// template void sgemm_warptiling<64, 64, 16, 32, 32, 2, 4, 4, 64>(
//     const torch::Tensor &, const torch::Tensor &, torch::Tensor &, float, float);


constexpr int WARPSIZE = 32;

// Host blocking helpers (row-major input -> blocked 4D layout)
void block_A_host(
    const float* A,      // row-major M x K
    float* Ablk)         // contiguous blocked: [BM][BK][TB_K][TB_M] (Transposed tiles)
{
    for (int bm = 0; bm < BM; ++bm) {
        for (int bk = 0; bk < BK; ++bk) {
            size_t block_index = (size_t)bm * BK + bk;
            float* dst_block = Ablk + block_index * (TB_M * TB_K);

            for (int i = 0; i < TB_M; ++i) {
                for (int j = 0; j < TB_K; ++j) {
                    int global_m = bm * TB_M + i;
                    int global_k = bk * TB_K + j;
                    // FIX: Store as [j][i] instead of [i][j] to achieve Column-Major tiles
                    dst_block[j * TB_M + i] = A[global_m * K + global_k];
                }
            }
        }
    }
}

void block_B_host(
    const float* B,      // row-major K x N
    float* Bblk)         // contiguous blocked: [BK][BN][TB_K][TB_N]
{
    for (int bk = 0; bk < BK; ++bk) {
        for (int bn = 0; bn < BN; ++bn) {
            size_t block_index = (size_t)bk * BN + bn;
            float* dst_block = Bblk + block_index * (TB_K * TB_N);

            for (int i = 0; i < TB_K; ++i) {
                for (int j = 0; j < TB_N; ++j) {
                    int global_k = bk * TB_K + i;
                    int global_n = bn * TB_N + j;
                    dst_block[i * TB_N + j] = B[global_k * N + global_n];
                }
            }
        }
    }
}

template <
    int TB_M, int TB_N, int TB_K, 
    int WM, int WN, int WNITER, 
    int TM, int TN, int NUM_THREADS
>
__global__ void __launch_bounds__(NUM_THREADS) blocked_gemm_kernel_templated(
    const float* __restrict__ Ablk,
    const float* __restrict__ Bblk,
    float* __restrict__ C,
    int M, int N, int K,
    int BM, int BN, int BK)
{
    // Derived constants calculated at compile-time
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // Block indices
    const int bm = blockIdx.x;
    const int bn = blockIdx.y;

    // Thread/Warp layout
    const int tid = threadIdx.x;
    const int warps_per_row = TB_N / WN;
    const int warp_idx = tid / WARPSIZE;
    const int warp_col = warp_idx % warps_per_row;
    const int warp_row = warp_idx / warps_per_row;

    const int thread_idx_in_warp = tid % WARPSIZE;
    const int thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const int thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    // Shared memory
    __shared__ float sA[TB_M * TB_K];
    __shared__ float sB[TB_K * TB_N];

    // Thread-local registers (compiler will now map these to RF)
    float accum[WMITER * TM * WNITER * TN] = {0.0f};
    float reg_m[WMITER * TM];
    float reg_n[WNITER * TN];

    for (int bk = 0; bk < BK; ++bk) {
        // 1. Vectorized Global -> Shared Load
        size_t a_block_base = (size_t)((bm * BK) + bk) * (TB_M * TB_K);
        size_t b_block_base = (size_t)((bk * BN) + bn) * (TB_K * TB_N);

        const float4* Ablk_vec = reinterpret_cast<const float4*>(&Ablk[a_block_base]);
        const float4* Bblk_vec = reinterpret_cast<const float4*>(&Bblk[b_block_base]);
        float4* sA_vec = reinterpret_cast<float4*>(sA);
        float4* sB_vec = reinterpret_cast<float4*>(sB);

        // Cooperative load for A
        #pragma unroll
        for (int idx = tid; idx < (TB_M * TB_K) / 4; idx += NUM_THREADS) {
            sA_vec[idx] = Ablk_vec[idx];
        }
        // Cooperative load for B
        #pragma unroll
        for (int idx = tid; idx < (TB_K * TB_N) / 4; idx += NUM_THREADS) {
            sB_vec[idx] = Bblk_vec[idx];
        }

        __syncthreads();

        // 2. Compute (Main Loop)
        #pragma unroll
        for (int kk = 0; kk < TB_K; ++kk) {
            
            // Load from Smem to Registers
            #pragma unroll
            for (int wm_idx = 0; wm_idx < WMITER; ++wm_idx) {
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    reg_m[wm_idx * TM + i] = sA[kk * TB_M + warp_row * WM + wm_idx * WSUBM + thread_row_in_warp * TM + i];
                }
            }

            #pragma unroll
            for (int wn_idx = 0; wn_idx < WNITER; ++wn_idx) {
                #pragma unroll
                for (int i = 0; i < TN; ++i) {
                    reg_n[wn_idx * TN + i] = sB[kk * TB_N + warp_col * WN + wn_idx * WSUBN + thread_col_in_warp * TN + i];
                }
            }

            // Outer product accumulation
            #pragma unroll
            for (int w_row = 0; w_row < WMITER; ++w_row) {
                #pragma unroll
                for (int w_col = 0; w_col < WNITER; ++w_col) {
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            accum[(w_row * TM + i) * (WNITER * TN) + (w_col * TN + j)] += 
                                reg_m[w_row * TM + i] * reg_n[w_col * TN + j];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // 3. Store Results to Global Memory
    float* C_ptr = C + (bm * TB_M + warp_row * WM) * N + bn * TB_N + warp_col * WN;

    #pragma unroll
    for (int w_row = 0; w_row < WMITER; ++w_row) {
        #pragma unroll
        for (int w_col = 0; w_col < WNITER; ++w_col) {
            float* tile_C = C_ptr + (w_row * WSUBM) * N + w_col * WSUBN;
            
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int row_idx = thread_row_in_warp * TM + i;
                int col_idx = thread_col_in_warp * TN;
                float4* row_C = reinterpret_cast<float4*>(&tile_C[row_idx * N + col_idx]);
                
                const int acc_base = (w_row * TM + i) * (WNITER * TN) + (w_col * TN);
                float4 result;
                result.x = accum[acc_base + 0];
                result.y = accum[acc_base + 1];
                result.z = accum[acc_base + 2];
                result.w = accum[acc_base + 3];
                
                row_C[0] = result;
            }
        }
    }
}

// CPU naive gemm for verification: C = A * B
void cpu_gemm_naive(const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

int main(int argc, char** argv) {
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else {
        std::cout << "Usage: ./gemm M N K\n";
        std::cout << "Note: Tile sizes are fixed at compile-time for template optimization.\n";
        std::cout << "Using default M=1024, N=1024, K=1024\n";
    }

    // Derived sizes
    BM = M / TB_M;
    BN = N / TB_N;
    BK = K / TB_K;

    // 1. Matrix Grid Fit
    assert(M % TB_M == 0 && "M must be divisible by TB_M");
    assert(N % TB_N == 0 && "N must be divisible by TB_N");
    assert(K % TB_K == 0 && "K must be divisible by TB_K");

    // 2. Tile-to-Warp Alignment
    // Your kernel uses constexpr WM=64, WN=64. TB sizes must be multiples of these.
    assert(TB_M % WM == 0 && "TB_M must be a multiple of WM (64)");
    assert(TB_N % WN == 0 && "TB_N must be a multiple of WN (64)");

    // 3. Warp Count Validation
    // The number of threads in the block must exactly match the number of warps needed
    // to cover the TB_M x TB_N area.
    int total_warps_needed = (TB_M / WM) * (TB_N / WN);
    assert(NUM_THREADS == total_warps_needed * WARPSIZE && 
        "NUM_THREADS must equal (TB_M/WM) * (TB_N/WN) * 32");

    // 4. Vectorization Requirements (float4)
    // TB_K and TB_N must be multiples of 4 to allow float4 loads from global memory.
    assert(TB_K % 4 == 0 && "TB_K must be a multiple of 4 for vectorized loads");
    assert(TB_N % 4 == 0 && "TB_N must be a multiple of 4 for vectorized loads");

    // 5. Shared Memory Capacity
    // Calculate based on runtime TB values and check against device limits.
    size_t smem_needed = (TB_M * TB_K + TB_K * TB_N) * sizeof(float);
    int max_smem_per_block = 0;
    cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    assert(smem_needed <= (size_t)max_smem_per_block && "Requested shared memory exceeds device limits");

    std::cout << "\n\nBlocked GEMM Compile Time (vec) end-to-end example\n";
    std::cout << "M=" << M << " K=" << K << " N=" << N << "\n";
    std::cout << "TB_M=" << TB_M << " TB_N=" << TB_N << " TB_K=" << TB_K << " TM=" << TM <<" TN=" << TN << "\n";

    // Allocate host row-major matrices
    size_t sizeA = (size_t)M * K;
    size_t sizeB = (size_t)K * N;
    size_t sizeC = (size_t)M * N;

    float* hA = (float*)malloc(sizeA * sizeof(float));
    float* hB = (float*)malloc(sizeB * sizeof(float));
    float* hC = (float*)malloc(sizeC * sizeof(float));
    float* hC_ref = (float*)malloc(sizeC * sizeof(float));

    // Fill random data
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) hA[i] = dist(rng);
    for (size_t i = 0; i < sizeB; ++i) hB[i] = dist(rng);

    // Allocate blocked host buffers
    size_t sizeAblk = (size_t)BM * BK * TB_M * TB_K;
    size_t sizeBblk = (size_t)BK * BN * TB_K * TB_N;

    float* hAblk = (float*)malloc(sizeAblk * sizeof(float));
    float* hBblk = (float*)malloc(sizeBblk * sizeof(float));

    // Block A and B on host
    block_A_host(hA, hAblk);
    block_B_host(hB, hBblk);

    // Device allocations
    float *dAblk = nullptr, *dBblk = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dAblk, sizeAblk * sizeof(float)));
    cudaCheck(cudaMalloc(&dBblk, sizeBblk * sizeof(float)));
    cudaCheck(cudaMalloc(&dC, sizeC * sizeof(float)));

    // Copy blocked tensors to device
    cudaCheck(cudaMemcpy(dAblk, hAblk, sizeAblk * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dBblk, hBblk, sizeBblk * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dC, 0, sizeC * sizeof(float)));

    // Launch kernel
    dim3 grid(BM, BN);
    dim3 block(NUM_THREADS); // one thread per TM output elements
    size_t smem_bytes = (TB_M * TB_K + TB_K * TB_N) * sizeof(float);

    std::cout << "Launching kernel grid(" << BM << "," << BN << ") block(" << NUM_THREADS << ") smem=" << smem_bytes << "\n";

    blocked_gemm_kernel_templated<
        TB_M, TB_N, TB_K, 
        WM, WN, WNITER, 
        TM, TN, NUM_THREADS
    ><<<grid, block>>>(dAblk, dBblk, dC, M, N, K, BM, BN, BK);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // Copy result back
    cudaCheck(cudaMemcpy(hC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    #if CPU_DEBUG
    // CPU reference
    cpu_gemm_naive(hA, hB, hC_ref);

    // Verify
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double d = fabs((double)hC_ref[i] - (double)hC[i]);
        sum_abs_diff += d;
        if (d > max_abs_diff) max_abs_diff = d;
    }

    std::cout << "Max absolute difference: " << max_abs_diff << "\n";
    std::cout << "Sum absolute difference: " << sum_abs_diff << "\n";

    const double eps = 1e-3; // tolerance (floating rounding)
    if (max_abs_diff < eps) {
        std::cout << "PASS: GPU result matches CPU reference within eps=" << eps << "\n";
    } else {
        std::cout << "FAIL: difference exceeds eps=" << eps << "\n";
    }

    #endif


    #if BENCHMARK
    // --------------------------
    // Timing (CUDA events)
    // --------------------------
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -----------------------------
    // Warm-up (5 iterations)
    // -----------------------------
    for (int i = 0; i < WARMUP; i++) {
        blocked_gemm_kernel_templated<
        TB_M, TB_N, TB_K, 
        WM, WN, WNITER, 
        TM, TN, NUM_THREADS><<<grid, block>>>(dAblk, dBblk, dC, M, N, K, BM, BN, BK);
    }
    cudaDeviceSynchronize();

    // -----------------------------
    // Benchmark (10 runs)
    // -----------------------------
    std::vector<float> times_ms;
    times_ms.reserve(ITER);

    float total_ms = 0.f, min_ms = 1e9f, max_ms = 0.f;

    for (int i = 0; i < ITER; i++) {
        cudaEventRecord(start);

        blocked_gemm_kernel_templated<
        TB_M, TB_N, TB_K, 
        WM, WN, WNITER, 
        TM, TN, NUM_THREADS><<<grid, block>>>(dAblk, dBblk, dC, M, N, K, BM, BN, BK);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);

        times_ms.push_back(ms);

        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    float mean_ms = total_ms / ITER;

    // ---- Compute sample standard deviation ----
    float var_ms = 0.f;
    for (float t : times_ms) {
        float diff = t - mean_ms;
        var_ms += diff * diff;
    }
    var_ms /= (ITER - 1);              // sample variance
    float std_ms = std::sqrt(var_ms);

    // ---- 95% confidence interval for time ----
    float stderr_ms = std_ms / std::sqrt((float)ITER);
    float ci95_ms = 1.96f * stderr_ms;

    // ---- GFLOP/s ----
    double flops = 2.0 * M * N * K;
    double gflops = flops / (mean_ms / 1000.0) / 1e9;

    // ---- Propagate CI to GFLOP/s ----
    double gflops_ci95 = gflops * (ci95_ms / mean_ms);

    // ---- Output ----
    std::cout << "---- Benchmark ----\n";
    std::cout << "Mean time: " << mean_ms << " ms\n";
    std::cout << "Min time:  " << min_ms << " ms\n";
    std::cout << "Max time:  " << max_ms << " ms\n";
    std::cout << "Stddev:    " << std_ms << " ms\n";

    std::cout << "Achieved:  " << gflops
            << " ± " << gflops_ci95
            << " GFLOP/s (95% CI)\n";

    #endif

    // Cleanup
    free(hA); free(hB); free(hC); free(hC_ref);
    free(hAblk); free(hBblk);
    cudaFree(dAblk); cudaFree(dBblk); cudaFree(dC);

    return 0;
}


