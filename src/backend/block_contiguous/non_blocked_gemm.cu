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

// Tile sizes (runtime)
constexpr int TB_M = 16;
constexpr int TB_N = 16;
constexpr int TB_K = 16;

// Matrix sizes (runtime)
int M = 1024;
int K = 1024;
int N = 1024;

// Derived block counts (runtime)
int BM, BK, BN;




// C is standard row-major M x N.
template <int TB_M_, int TB_N_, int TB_K_>
__global__ void gemm_kernel(
    const float* Ablk,
    const float* Bblk,
    float* C,
    int M_, int N_, int K_,
    int BM_, int BN_, int BK_)
{
    const int bm = blockIdx.x; // 0..BM-1
    const int bn = blockIdx.y; // 0..BN-1

    // Each thread computes one element in TB_M x TB_N tile
    const int tid = threadIdx.x;
    const int tm = tid / TB_N_;
    const int tn = tid % TB_N_;

    if (tm >= TB_M_ || tn >= TB_N_) return;

    // Shared memory for tiles

    __shared__ float sA[TB_M_ * TB_K_];
    __shared__ float sB[TB_K * TB_N];

    float accum = 0.0f;

    // Loop over BK tiles
    for (int bk = 0; bk < BK_; ++bk) {
        
        // GLOBAL TILE ORIGIN IN NON-BLOCKED MATRICES
        int global_row_A = bm * TB_M_;         // starting row of tile in A
        int global_col_A = bk * TB_K_;         // starting col of tile in A

        int global_row_B = bk * TB_K_;         // starting row of tile in B
        int global_col_B = bn * TB_N_;         // starting col of tile in B

        // cooperative load of A tile: TB_M x TB_K
        for (int idx = tid; idx < TB_M_ * TB_K_; idx += blockDim.x) {
            int i = idx / TB_K_;   // local row tid-> 0,1,..., 63, 8x4
            int j = idx % TB_K_;   // local col

            int row = global_row_A + i;
            int col = global_col_A + j;


            if (row < M_ && col < K_)
                sA[i * TB_K_ + j] = Ablk[row * K_ + col];   // row-major A
            else
                sA[i * TB_K_ + j] = 0.0f;
        }

        // cooperative load of B tile: TB_K × TB_N
        for (int idx = tid; idx < TB_K_ * TB_N_; idx += blockDim.x) {
            int i = idx / TB_N_;   // local row
            int j = idx % TB_N_;   // local col

            int row = global_row_B + i;
            int col = global_col_B + j;

            if (row < K_ && col < N_)
                sB[i * TB_N_ + j] = Bblk[row * N_ + col];   // row-major B
            else
                sB[i * TB_N_ + j] = 0.0f;
        }

        __syncthreads();

        // Compute partial GEMM for this tile
        for (int kk = 0; kk < TB_K_; ++kk) {
            float a = sA[tm * TB_K_ + kk];
            float b = sB[kk * TB_N_ + tn];
            accum += a * b;
        }

        __syncthreads();
    }

    // Write C tile result
    int row = bm * TB_M_ + tm;
    int col = bn * TB_N_ + tn;

    if (row < M_ && col < N_) {
        C[row * N_ + col] = accum;
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

    std::cout << "Non-Blocked GEMM (Naive)\n";
    std::cout << "M=" << M << " K=" << K << " N=" << N << "\n";
    std::cout << "TB_M=" << TB_M << " TB_N=" << TB_N << " TB_K=" << TB_K << "\n";

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

    // Device allocations
    float *dAblk = nullptr, *dBblk = nullptr, *dC = nullptr;
    cudaCheck(cudaMalloc(&dAblk, sizeA * sizeof(float)));
    cudaCheck(cudaMalloc(&dBblk, sizeB * sizeof(float)));
    cudaCheck(cudaMalloc(&dC, sizeC * sizeof(float)));

    // Copy blocked tensors to device
    cudaCheck(cudaMemcpy(dAblk, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dBblk, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dC, 0, sizeC * sizeof(float)));

    // Launch kernel
    dim3 grid(BM, BN);
    dim3 block(TB_M * TB_N); // one thread per output element
    size_t smem_bytes = (TB_M * TB_K + TB_K * TB_N) * sizeof(float);

    std::cout << "Launching kernel grid(" << BM << "," << BN << ") block(" << (TB_M*TB_N) << ") smem=" << smem_bytes << "\n";

    gemm_kernel<TB_M, TB_N, TB_K><<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K, BM, BN, BK);
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
    // Warm-up 
    // -----------------------------
    for (int i = 0; i < WARMUP; i++) {
        gemm_kernel<TB_M, TB_N, TB_K><<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K, BM, BN, BK);
    }
    cudaDeviceSynchronize();

    // -----------------------------
    // Benchmark
    // -----------------------------
    std::vector<float> times_ms;
    times_ms.reserve(ITER);

    float total_ms = 0.f, min_ms = 1e9f, max_ms = 0.f;

    std::cout << "-----BEGIN\n";
    for (int i = 0; i < ITER; i++) {
        cudaEventRecord(start);

        gemm_kernel<TB_M, TB_N, TB_K><<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K, BM, BN, BK);

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
    free(hA); free(hB); free(hC); free(hC_ref);
    cudaFree(dAblk); cudaFree(dBblk); cudaFree(dC);

    return 0;
}


