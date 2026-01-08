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


// simple CUDA error-check macro
inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// Tile sizes (runtime)
int TB_M = 128;
int TB_N = 128;
int TB_K = 16;

// Matrix sizes (runtime)
int M = 4096;
int K = 1024;
int N = 4096;

constexpr int TM = 8;
constexpr int TN = 4;

constexpr int WARPSIZE = 32;
constexpr int WM = 64;
constexpr int WN = 64;
constexpr int WNITER= 4;
constexpr int NUM_THREADS = 128;

// Derived block counts (runtime)
int BM, BK, BN;




// C is standard row-major M x N.
__global__ void __launch_bounds__(NUM_THREADS) gemm_kernel(
    const float* Ablk,
    const float* Bblk,
    float* C,
    int M_, int N_, int K_,
    int TB_M_, int TB_N_, int TB_K_,
    int BM_, int BN_, int BK_)
{
    // Which block of C this is:
    const int bm = blockIdx.x; // 0..BM-1
    const int bn = blockIdx.y; // 0..BN-1

    // each thread computes TM output elements within TB_M x TB_N
    const int tid = threadIdx.x;
    const int warps_per_row = TB_N_/WN;

    const int warp_idx = tid/WARPSIZE;
    const int warp_col = warp_idx % warps_per_row;
    const int warp_row = warp_idx / warps_per_row;

    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    static_assert(WSUBN % TN == 0, "Sub-tile width must be divisible by TN");
    static_assert(WARPSIZE % (WSUBN / TN) == 0, "Warp must be able to distribute evenly across TN columns");
    static_assert(TN % 4 == 0, "TN must be 4 for float4 vectorized stores to C");

    const int thread_idx_in_warp = threadIdx.x % WARPSIZE;
    const int thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const int thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);


    // How many threads are there in one row of the tile?
    const int threads_per_row = TB_N_ / TN;
    const int tm = tid / threads_per_row; // 0..TB_M-1
    const int tn = tid % threads_per_row; // 0..TB_N-1

    if (tm >= TB_M_ || tn >= TB_N_) return;

    // Shared memory for tiles
    extern __shared__ float smem[]; // size = TB_M*TB_K + TB_K*TB_N
    float* sA = smem;                       // TB_M * TB_K
    float* sB = smem + TB_M_ * TB_K_;        // TB_K * TB_N

    // Allocate thread-local cache for results in registerfile
    float accum[WMITER * TM * WNITER * TN] = {0.0f};
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    // Loop over bk blocks
    for (int bk = 0; bk < BK_; ++bk) {

        // GLOBAL TILE ORIGIN IN NON-BLOCKED MATRICES
        int global_row_A = bm * TB_M_;         // starting row of tile in A
        int global_col_A = bk * TB_K_;         // starting col of tile in A

        int global_row_B = bk * TB_K_;         // starting row of tile in B
        int global_col_B = bn * TB_N_;         // starting col of tile in B

        // 1. Cast Shared Memory pointers to float4
        float4* sA_vec = reinterpret_cast<float4*>(sA);
        float4* sB_vec = reinterpret_cast<float4*>(sB);

        // We assume TB_K is a multiple of 4.
        int num_vec_A = (TB_M_ * TB_K_) / 4;

        // cooperative load of A tile: TB_M x TB_K
        for (int idx = tid; idx < num_vec_A; idx += blockDim.x) {
            int k_local = idx / (TB_M_ / 4); // row in the transposed matrix (K-dim)
            int m_vec   = idx % (TB_M_ / 4); // col in the transposed matrix (M-dim)
            
            // Global coordinates in the TRANSPOSED matrix
            int global_k = bk * TB_K_ + k_local;
            int global_m = bm * TB_M_ + (m_vec * 4);

            if (global_k < K_ && global_m < M_) {
                // Linear load is now perfectly aligned with sA's required layout
                sA_vec[idx] = *reinterpret_cast<const float4*>(&Ablk[global_k * M_ + global_m]);
            } else {
                sA_vec[idx] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
        
        int num_vec_B = (TB_K_ * TB_N_) / 4;

        // cooperative load of B tile: TB_K × TB_N
        for (int idx = tid; idx < num_vec_B; idx += blockDim.x) {
            int i = idx / (TB_N_/4);   // local row
            int j = idx % (TB_N_/4);   // local col
            j*=4;

            int row = global_row_B + i;
            int col = global_col_B + j;

            const float4* Bblk_vec = reinterpret_cast<const float4*>(&Bblk[row * N_ + col]);

            if (row < K_ && col < N_)
                sB_vec[idx] = *Bblk_vec;   // row-major B
            else
                sB_vec[idx] = {0.0f, 0.0f, 0.0f, 0.0f};
        }

        __syncthreads();

        // compute partial product for this thread's (tm,tn)
        for (int kk = 0; kk < TB_K_; ++kk) {
            #pragma unroll
            for (int wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx){
                #pragma unroll
                for (int i = 0;i < TM;  i++){
                    register_m[wsub_row_idx * TM + i] =  sA[(kk * TB_M_) + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i];
                }
            }
            #pragma unroll
            for (int wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
            {
                #pragma unroll
                for (int i = 0;i < TN; i++){
                    register_n[wsub_col_idx * TN + i] = sB[(kk * TB_N_) + warp_col * WN + wsub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i];
                }
            }

            #pragma unroll
            for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
            {
                #pragma unroll
                for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
                {
                    // Each thread calculates TM x TN outputs
                    #pragma unroll
                    for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
                    {
                        #pragma unroll
                        for (int res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                        {
                            accum[(wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                       (wsub_col_idx * TN) + res_idx_n] +=
                            register_m[wsub_row_idx * TM + res_idx_m] *
                            register_n[wsub_col_idx * TN + res_idx_n];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }
 

    C+= (bm * TB_M_ + warp_row * WM) * N_ + bn * TB_N_ + warp_col * WN;
    #pragma unroll
    for (uint wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx)
    {
        #pragma unroll
        for (uint wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
        {
            float *matrix_c_interim = C + (wsub_row_idx * WSUBM) * N_ +
                                      wsub_col_idx * WSUBN;

            #pragma unroll
            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1)
            {
                #pragma unroll
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4)
                {
                    float4 tmp_c = reinterpret_cast<float4 *>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * N_ +
                                          thread_col_in_warp * TN + res_idx_n])[0];

                    const int res_idx = (wsub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                        wsub_col_idx * TN + res_idx_n;
                    tmp_c.x = accum[res_idx + 0];
                    tmp_c.y = accum[res_idx + 1];
                    tmp_c.z = accum[res_idx + 2];
                    tmp_c.w = accum[res_idx + 3];

                    reinterpret_cast<float4 *>(
                        &matrix_c_interim[(thread_row_in_warp * TM + res_idx_m) * N_ +
                                          thread_col_in_warp * TN + res_idx_n])[0] = tmp_c;
                }
            }
        }
    }
}

void transpose_host(const float* src, float* dst, size_t rows, size_t cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // src is [rows][cols], dst is [cols][rows]
            // src index: i * cols + j
            // dst index: j * rows + i
            dst[j * rows + i] = src[i * cols + j];
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
    if (argc == 7) {
        M    = std::atoi(argv[1]);
        N    = std::atoi(argv[2]);
        K    = std::atoi(argv[3]);
        TB_M = std::atoi(argv[4]);
        TB_N = std::atoi(argv[5]);
        TB_K = std::atoi(argv[6]);
    } else {
        std::cout << "Usage: ./gemm M N K TB_M TB_N TB_K\n";
        std::cout << "Using default values.\n";
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

    std::cout << "\n\nNon-blocked GEMM (vec) end-to-end example\n";
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

    // Transpose A.
    float* hA_t = (float*)malloc(sizeA * sizeof(float));
    transpose_host(hA, hA_t, M, K);

    // Device allocations
    float *dAblk = nullptr, *dBblk = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dAblk, sizeA * sizeof(float)));
    cudaCheck(cudaMalloc(&dBblk, sizeB * sizeof(float)));
    cudaCheck(cudaMalloc(&dC, sizeC * sizeof(float)));

    // Copy blocked tensors to device
    cudaCheck(cudaMemcpy(dAblk, hA_t, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dBblk, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dC, 0, sizeC * sizeof(float)));

    // Launch kernel
    dim3 grid(BM, BN);
    dim3 block(NUM_THREADS); // one thread per TM output elements
    size_t smem_bytes = (TB_M * TB_K + TB_K * TB_N) * sizeof(float);

    std::cout << "Launching kernel grid(" << BM << "," << BN << ") block(" << NUM_THREADS << ") smem=" << smem_bytes << "\n";

    gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K, TB_M, TB_N, TB_K, BM, BN, BK);
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
        gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K, TB_M, TB_N, TB_K, BM, BN, BK);
    }
    cudaDeviceSynchronize();

    // -----------------------------
    // Benchmark (10 runs)
    // -----------------------------
    std::vector<float> times_ms;
    times_ms.reserve(ITER);

    float total_ms = 0.f, min_ms = 1e9f, max_ms = 0.f;

    std::cout << "-----BEGIN\n";
    for (int i = 0; i < ITER; i++) {
        cudaEventRecord(start);

        gemm_kernel<<<grid, block, smem_bytes>>>(
            dAblk, dBblk, dC,
            M, N, K,
            TB_M, TB_N, TB_K,
            BM, BN, BK
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);

        times_ms.push_back(ms);

        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }
    std::cout << "-----END\n";

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
    free(hA); free(hB); free(hC); free(hC_ref); free(hA_t);
    cudaFree(dAblk); cudaFree(dBblk); cudaFree(dC);

    return 0;
}


