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
#define BENCHMARK 0
#endif

#if BENCHMARK
float WARMUP=3;
float ITER=10;
#endif


// simple CUDA error-check macro
inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// Tile sizes (runtime)
int TB_M = 16;
int TB_N = 16;
int TB_K = 16;

// Matrix sizes (runtime)
int M = 4096;
int K = 4096;
int N = 4096;

// Derived block counts (runtime)
int BM, BK, BN;




// C is standard row-major M x N.
__global__ void gemm_kernel(
    const float* Ablk,
    const float* Bblk,
    float* C,
    int M_, int N_, int K_,
    int TB_M_, int TB_N_, int TB_K_,
    int BM_, int BN_, int BK_)
{
    const int bm = blockIdx.x; // 0..BM-1
    const int bn = blockIdx.y; // 0..BN-1

    // Each thread computes one element in TB_M x TB_N tile
    const int tid = threadIdx.x;
    const int tm = tid / TB_N_;
    const int tn = tid % TB_N_;

    if (tm >= TB_M_ || tn >= TB_N_) return;

    // Shared memory
    extern __shared__ float smem[];
    float* sA = smem;                       // TB_M * TB_K
    float* sB = smem + TB_M_ * TB_K_;         // TB_K * TB_N

    float accum = 0.0f;

    // Loop over BK tiles
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
            int i = idx / (TB_K_/4);   // local row tid-> 0,1,..., 63, 8x4
            int j = idx % (TB_K_/4);   // local col
            j*=4;

            int row = global_row_A + i;
            int col = global_col_A + j;
            
            const float4* Ablk_vec = reinterpret_cast<const float4*>(&Ablk[row * K_ + col]);

            if (row < M_ && col < K_)
                sA_vec[idx] = *Ablk_vec;   // row-major A
            else
                sA_vec[idx] = {0.0f, 0.0f, 0.0f, 0.0f};
            

            // if (row < M_ && col < K_)
            //     sA[i * TB_K_ + j] = Ablk[row * K_ + col];   // row-major A
            // else
            //     sA[i * TB_K_ + j] = 0.0f;
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

    std::cout << "Non-Blocked GEMM (vec) end-to-end example\n";
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
    float total_ms = 0.f, min_ms = 1e9, max_ms = 0.f;

    for (int i = 0; i < ITER; i++) {
        cudaEventRecord(start);

        gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K, TB_M, TB_N, TB_K, BM, BN, BK);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    float avg_ms = total_ms / ITER;

    std::cout << "---- Benchmark ----\n";
    std::cout << "Avg time: " << avg_ms << " ms\n";
    std::cout << "Min time: " << min_ms << " ms\n";
    std::cout << "Max time: " << max_ms << " ms\n";

    double gflops = (2.0 * M * N * K) / (avg_ms/1000.0) / 1e9;
    std::cout << "Achieved: " << gflops << " GFLOP/s\n";

    #endif

    // Cleanup
    free(hA); free(hB); free(hC); free(hC_ref);
    cudaFree(dAblk); cudaFree(dBblk); cudaFree(dC);

    return 0;
}


