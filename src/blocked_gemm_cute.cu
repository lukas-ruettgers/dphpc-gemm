// gemm_blocked_example.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>
#include <cute/layout.hpp>

// simple CUDA error-check macro
inline void cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

// Tile sizes (tune as desired)
constexpr int TB_M = 16;
constexpr int TB_N = 16;
constexpr int TB_K = 16;

// Matrix sizes (must be multiples of TB sizes in this simple demo)
constexpr int M = 128;   // rows of A and C
constexpr int K = 128;   // cols of A, rows of B
constexpr int N = 128;   // cols of B and C

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

// Device kernel: expects Ablk and Bblk in blocked layout described above.
// C is standard row-major M x N.
__global__ void blocked_gemm_kernel(
    const float* __restrict__ Ablk,  // [BM][BK][TB_M][TB_K] flattened
    const float* __restrict__ Bblk,  // [BK][BN][TB_K][TB_N] flattened
    float* __restrict__ C,           // row-major M x N
    int M_, int N_, int K_)
{
    // Which block of C this is:
    const int bm = blockIdx.x; // 0..BM-1
    const int bn = blockIdx.y; // 0..BN-1

    // each thread computes one output element within TB_M x TB_N
    const int tid = threadIdx.x;
    const int tm = tid / TB_N; // 0..TB_M-1
    const int tn = tid % TB_N; // 0..TB_N-1

    if (tm >= TB_M || tn >= TB_N) return;

    // Shared memory for tiles
    extern __shared__ float smem[]; // size = TB_M*TB_K + TB_K*TB_N
    float* sA = smem;                       // TB_M * TB_K
    float* sB = smem + TB_M * TB_K;        // TB_K * TB_N

    float accum = 0.0f;

    // Loop over bk blocks
    for (int bk = 0; bk < BK; ++bk) {
        // Load A block [TB_M x TB_K] from global Ablk to sA
        // global address: Ablk[ ((bm * BK) + bk) * (TB_M*TB_K) + i*TB_K + j ]
        size_t a_block_base = (size_t)((bm * BK) + bk) * (TB_M * TB_K);
        size_t b_block_base = (size_t)((bk * BN) + bn) * (TB_K * TB_N);

        // cooperative load
        for (int idx = tid; idx < TB_M * TB_K; idx += blockDim.x) {
            int i = idx / TB_K;
            int j = idx % TB_K;
            sA[i * TB_K + j] = Ablk[a_block_base + i * TB_K + j];
        }

        for (int idx = tid; idx < TB_K * TB_N; idx += blockDim.x) {
            int i = idx / TB_N;
            int j = idx % TB_N;
            sB[i * TB_N + j] = Bblk[b_block_base + i * TB_N + j];
        }

        __syncthreads();

        // compute partial product for this thread's (tm,tn)
        for (int kk = 0; kk < TB_K; ++kk) {
            float a = sA[tm * TB_K + kk];
            float b = sB[kk * TB_N + tn];
            accum += a * b;
        }

        __syncthreads();
    }

    // Write to C
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
    std::cout << "Blocked GEMM end-to-end example\n";
    std::cout << "M=" << M << " K=" << K << " N=" << N << "\n";
    std::cout << "TB_M=" << TB_M << " TB_N=" << TB_N << " TB_K=" << TB_K << "\n";

    int m = 8, n = 4, k = 2;
    auto a = make_layout(make_shape(m, k), make_stride(k, 1));
    print_layout(a);

    return 0;

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
    dim3 block(TB_M * TB_N); // one thread per output element
    size_t smem_bytes = (TB_M * TB_K + TB_K * TB_N) * sizeof(float);

    std::cout << "Launching kernel grid(" << BM << "," << BN << ") block(" << (TB_M*TB_N) << ") smem=" << smem_bytes << "\n";

    blocked_gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K);
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

    // Cleanup
    free(hA); free(hB); free(hC); free(hC_ref);
    free(hAblk); free(hBblk);
    cudaFree(dAblk); cudaFree(dBblk); cudaFree(dC);

    return 0;
}


