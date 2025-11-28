// gemm_blocked_example.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>

// simple CUDA error-check macro
inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// Tile sizes (tune as desired)
constexpr int TB_M = 32;
constexpr int TB_N = 32;
constexpr int TB_K = 32;

// Matrix sizes (must be multiples of TB sizes in this simple demo)
constexpr int M = 1024;   // rows of A and C
constexpr int K = 1024;   // cols of A, rows of B
constexpr int N = 1024;   // cols of B and C

// Derived block counts
constexpr int BM = M / TB_M;
constexpr int BK = K / TB_K;
constexpr int BN = N / TB_N;

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

// C is standard row-major M x N.
__global__ void gemm_kernel(
    const float* __restrict__ Ablk,  // now interpreted as row-major A[M x K]
    const float* __restrict__ Bblk,  // now interpreted as row-major B[K x N]
    float* __restrict__ C,           // row-major M x N
    int M_, int N_, int K_)
{
    const int bm = blockIdx.x; // 0..BM-1
    const int bn = blockIdx.y; // 0..BN-1

    // Each thread computes one element in TB_M x TB_N tile
    const int tid = threadIdx.x;
    const int tm = tid / TB_N;
    const int tn = tid % TB_N;

    if (tm >= TB_M || tn >= TB_N) return;

    // Shared memory
    extern __shared__ float smem[];
    float* sA = smem;                       // TB_M * TB_K
    float* sB = smem + TB_M * TB_K;         // TB_K * TB_N

    float accum = 0.0f;

    // Loop over BK tiles
    for (int bk = 0; bk < BK; ++bk) {
        
        // GLOBAL TILE ORIGIN IN NON-BLOCKED MATRICES
        int global_row_A = bm * TB_M;         // starting row of tile in A
        int global_col_A = bk * TB_K;         // starting col of tile in A

        int global_row_B = bk * TB_K;         // starting row of tile in B
        int global_col_B = bn * TB_N;         // starting col of tile in B

        // cooperative load of A tile: TB_M x TB_K
        for (int idx = tid; idx < TB_M * TB_K; idx += blockDim.x) {
            int i = idx / TB_K;   // local row
            int j = idx % TB_K;   // local col

            int row = global_row_A + i;
            int col = global_col_A + j;

            if (row < M_ && col < K_)
                sA[i * TB_K + j] = Ablk[row * K_ + col];   // row-major A
            else
                sA[i * TB_K + j] = 0.0f;
        }

        // cooperative load of B tile: TB_K × TB_N
        for (int idx = tid; idx < TB_K * TB_N; idx += blockDim.x) {
            int i = idx / TB_N;   // local row
            int j = idx % TB_N;   // local col

            int row = global_row_B + i;
            int col = global_col_B + j;

            if (row < K_ && col < N_)
                sB[i * TB_N + j] = Bblk[row * N_ + col];   // row-major B
            else
                sB[i * TB_N + j] = 0.0f;
        }

        __syncthreads();

        // Compute partial GEMM for this tile
        for (int kk = 0; kk < TB_K; ++kk) {
            float a = sA[tm * TB_K + kk];
            float b = sB[kk * TB_N + tn];
            accum += a * b;
        }

        __syncthreads();
    }

    // Write C tile result
    int row = bm * TB_M + tm;
    int col = bn * TB_N + tn;

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

int main() {
    std::cout << "Non-Blocked GEMM end-to-end example\n";
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

    gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());

    // Copy result back
    cudaCheck(cudaMemcpy(hC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

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
    for (int i = 0; i < 5; i++) {
        gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K);
    }
    cudaDeviceSynchronize();

    // -----------------------------
    // Benchmark (10 runs)
    // -----------------------------
    float total_ms = 0.f, min_ms = 1e9, max_ms = 0.f;

    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);

        gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        total_ms += ms;
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
    }

    float avg_ms = total_ms / 10.0f;

    std::cout << "---- Benchmark ----\n";
    std::cout << "Avg time: " << avg_ms << " ms\n";
    std::cout << "Min time: " << min_ms << " ms\n";
    std::cout << "Max time: " << max_ms << " ms\n";

    double gflops = (2.0 * M * N * K) / (avg_ms/1000.0) / 1e9;
    std::cout << "Achieved: " << gflops << " GFLOP/s\n";


    // Cleanup
    free(hA); free(hB); free(hC); free(hC_ref);
    cudaFree(dAblk); cudaFree(dBblk); cudaFree(dC);

    return 0;
}


