// gemm_blocked_example.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>

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

// Matrix sizes (runtime)
int M = 1024;
int K = 1024;
int N = 1024;

// Tile sizes (runtime)
constexpr int TB_M = 128;
constexpr int TB_N = 128;
constexpr int TB_K = 16;

constexpr int TM = 8;
constexpr int TN = 8;

// Derived block counts (runtime)
int BM, BK, BN;

// Host blocking helpers (row-major input -> blocked 4D layout)
void block_A_host(
    const float* A,      // row-major M x K
    float* Ablk)         // contiguous blocked: [BM][BK][TB_M][TB_K]
{
    for (int bm = 0; bm < BM; ++bm) {
        for (int bk = 0; bk < BK; ++bk) {
            // pointer to destination block
            size_t block_index = (size_t)bm * BK + bk;
            float* dst_block = Ablk + block_index * (TB_M * TB_K);

            for (int i = 0; i < TB_M; ++i) {
                for (int j = 0; j < TB_K; ++j) {
                    int global_m = bm * TB_M + i;
                    int global_k = bk * TB_K + j;
                    dst_block[i * TB_K + j] = A[global_m * K + global_k];
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

// Device kernel: expects Ablk and Bblk in blocked layout described above.
// C is standard row-major M x N.
template<
    int TB_M_, int TB_N_, int TB_K_,
    int TM, int TN
>
__global__ void blocked_gemm_kernel(
    const float* Ablk,
    const float* Bblk,
    float* C,
    int M_, int N_, int K_,
    int BM_, int BN_, int BK_)
{
    // Which block of C this is:
    const int bm = blockIdx.x; // 0..BM-1
    const int bn = blockIdx.y; // 0..BN-1

    // each thread computes TM output elements within TB_M x TB_N
    const int tid = threadIdx.x;
    // How many threads are there in one row of the tile?
    const int threads_per_row = TB_N_ / TN;

    const int tm = tid / threads_per_row; // 0..TB_M-1
    const int tn = tid % threads_per_row; // 0..TB_N-1

    if (tm >= TB_M_ || tn >= TB_N_) return;

    // Shared memory for tiles
    __shared__ float sA[TB_M * TB_K];
    __shared__ float sB[TB_K * TB_N];
    
    // Allocate thread-local cache for results in registerfile
    float accum[TM*TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};

    // Loop over bk blocks
    for (int bk = 0; bk < BK_; ++bk) {

        // Load A block [TB_M x TB_K] from global Ablk to sA
        // global address: Ablk[ ((bm * BK) + bk) * (TB_M*TB_K) + i*TB_K + j ]
        size_t a_block_base = (size_t)((bm * BK_) + bk) * (TB_M_ * TB_K_);
        size_t b_block_base = (size_t)((bk * BN_) + bn) * (TB_K_ * TB_N_);

        // 1. Cast Shared Memory pointers to float4
        float4* sA_vec = reinterpret_cast<float4*>(sA);
        float4* sB_vec = reinterpret_cast<float4*>(sB);

        // 2. Cast Global Memory pointers to float4
        // Note: We compute the base address in 'float' offset first, then cast to float4*
        const float4* Ablk_vec = reinterpret_cast<const float4*>(&Ablk[a_block_base]);
        const float4* Bblk_vec = reinterpret_cast<const float4*>(&Bblk[b_block_base]);

        // 3. Load A (Vectorized)
        // We assume TB_K is a multiple of 4.
        int num_vec_A = (TB_M_ * TB_K_) / 4; 
        
        for (int idx = tid; idx < num_vec_A; idx += blockDim.x) {
            // The copy is now a single instruction moving 128 bits
            sA_vec[idx] = Ablk_vec[idx];
        }

        // 4. Load B (Vectorized)
        // We assume TB_N is a multiple of 4.
        int num_vec_B = (TB_K_ * TB_N_) / 4;

        for (int idx = tid; idx < num_vec_B; idx += blockDim.x) {
            sB_vec[idx] = Bblk_vec[idx];
        }

        __syncthreads();

        // compute partial product for this thread's (tm,tn)
        for (int kk = 0; kk < TB_K_; ++kk) {
            #pragma unroll
            for (int i = 0;i < TM;  i++){
                register_m[i] =  sA[(tm * TM + i)*TB_K_ + kk];
            }
            #pragma unroll
            for (int i = 0;i < TN; i++){
                register_n[i] = sB[kk*TB_N_ + tn*TN +i];
            }
            // Each thread calculates TM x TN outputs
            #pragma unroll
            for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
            {
                #pragma unroll
                for (int res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                {
                    accum[res_idx_m * TN + res_idx_n] +=
                        register_m[res_idx_m] * register_n[res_idx_n];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
    {
        const int row = bm * TB_M_ + tm * TM + res_idx_m;
#pragma unroll
        for (int res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
        {
            const int col = bn * TB_N_ + tn * TN + res_idx_n;
            // Add this check!
            if (row < M_ && col < N_) {
                const int c_idx = row * N_ + col;
                C[c_idx] = accum[res_idx_m * TN + res_idx_n];
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

    std::cout << "\n\nBlocked GEMM (2d vec) end-to-end example\n";
    std::cout << "M=" << M << " K=" << K << " N=" << N << "\n";
    std::cout << "TB_M=" << TB_M << " TB_N=" << TB_N << " TB_K=" << TB_K << " TM=" << TM << "\n";

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
    dim3 block((TB_M/TM) * (TB_N/TN)); // one thread per TM output elements
    size_t smem_bytes = (TB_M * TB_K + TB_K * TB_N) * sizeof(float);

    std::cout << "Launching kernel grid(" << BM << "," << BN << ") block(" << ((TB_M/TM) * (TB_N/TN)) << ") smem=" << smem_bytes << "\n";

    blocked_gemm_kernel<TB_M, TB_N, TB_K, TM, TN><<<grid, block >>>(dAblk, dBblk, dC, M, N, K, BM, BN, BK);
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
        blocked_gemm_kernel<TB_M, TB_N, TB_K, TM, TN><<<grid, block >>>(dAblk, dBblk, dC, M, N, K, BM, BN, BK);
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

        blocked_gemm_kernel<TB_M, TB_N, TB_K, TM, TN><<<grid, block >>>(dAblk, dBblk, dC, M, N, K, BM, BN, BK);

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


