// gemm_blocked_example.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

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

// constexpr int TB_M = 2;
// constexpr int TB_N = 2;
// constexpr int TB_K = 2;

// // Matrix sizes (must be multiples of TB sizes in this simple demo)
// constexpr int M = 4;   // rows of A and C
// constexpr int K = 4;   // cols of A, rows of B
// constexpr int N = 4;   // cols of B and C

// Derived block counts
constexpr int BM = M / TB_M;
constexpr int BK = K / TB_K;
constexpr int BN = N / TB_N;

template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}


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
__global__ void cute_blocked_gemm_kernel(
    const float* __restrict__ Ablk,  // [BM][BK][TB_M][TB_K] flattened
    const float* __restrict__ Bblk,  // [BK][BN][TB_K][TB_N] flattened
    float* __restrict__ C,           // row-major M x N
    int M_, int N_, int K_)
{
    using namespace cute;

    // Which block of C this is:
    const int bm = blockIdx.x; // 0..BM-1
    const int bn = blockIdx.y; // 0..BN-1

    // Thread identification - each thread computes one element in TB_M x TB_N tile
    const int tid = threadIdx.x;
    const int tm = tid / TB_N; // 0..TB_M-1
    const int tn = tid % TB_N; // 0..TB_N-1

    if (tm >= TB_M || tn >= TB_N) return;

    // Shared memory allocation
    extern __shared__ float smem[]; // size = TB_M*TB_K + TB_K*TB_N
    float* smem_A = smem;                       // TB_M * TB_K
    float* smem_B = smem + TB_M * TB_K;        // TB_K * TB_N


    // Create CUTE tensors for shared memory
    Layout sA_layout = make_layout(make_shape(TB_M, TB_K), make_stride(TB_K, 1));
    Layout sB_layout = make_layout(make_shape(TB_K, TB_N), make_stride(TB_N, 1));
    Tensor sA = make_tensor(make_smem_ptr(smem_A), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem_B), sB_layout);

    auto Ablk_layout = make_layout(
        make_shape(BM, BK, TB_M, TB_K),
        make_stride(K*TB_M, TB_M*TB_K, TB_K, 1)
    );
    auto Bblk_layout = make_layout(
        make_shape(BK, BN, TB_K, TB_N),
        make_stride(N*TB_K, TB_K*TB_N, TB_N, 1)
    );

    // C layout: row-major M x N
    auto C_layout = make_layout(make_shape(M_, N_), make_stride(N_, 1));

    // Create CUTE tensors for global memory
    Tensor gAblk = make_tensor(make_gmem_ptr(Ablk), Ablk_layout);
    Tensor gBblk = make_tensor(make_gmem_ptr(Bblk), Bblk_layout);
    Tensor gC = make_tensor(make_gmem_ptr(C), C_layout);

    // Create register accumulator for this thread's output element
    // We'll manually manage the accumulation since each thread computes one element
    float accum = 0.0f;

    // Loop over bk blocks
    for (int bk = 0; bk < BK; ++bk) {
        // Extract the current blocks using CUTE slicing
        // Get A block: gAblk[bm, bk, :, :] - a TB_M x TB_K tile
        auto gA_block = gAblk(bm, bk, _, _);
        // Get B block: gBblk[bk, bn, :, :] - a TB_K x TB_N tile
        auto gB_block = gBblk(bk, bn, _, _);
        
        // Load A tile
        for (int idx = tid; idx < TB_M * TB_K; idx += blockDim.x) {
            int i = idx / TB_K;
            int j = idx % TB_K;
            sA(i, j) = gA_block(i, j);
        }

        // Load B tile  
        for (int idx = tid; idx < TB_K * TB_N; idx += blockDim.x) {
            int i = idx / TB_N;
            int j = idx % TB_N;
            sB(i, j) = gB_block(i, j);
        }

        __syncthreads();

        // Compute partial product using CUTE tensor access
        // This thread computes: accum += sum_over_kk(sA[tm, kk] * sB[kk, tn])
        for (int kk = 0; kk < TB_K; ++kk) {
            accum += sA(tm, kk) * sB(kk, tn);
        }

        __syncthreads();
    }

    // Write result using CUTE tensor access
    int row = bm * TB_M + tm;
    int col = bn * TB_N + tn;
    if (row < M_ && col < N_) {
        gC(row, col) = accum;
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

void example_and_quit() {
    constexpr int m = 24, k = 8;
    constexpr int tb_m = 12, tb_k = 4;
    constexpr int warp_m = 4, warp_k = 2;

    Layout a = make_layout(make_shape(m, k), make_stride(k, 1)); // Row major
    print2D(a);
    print("\n");

    auto a_blk1_shape = make_shape(make_shape(tb_m, m / tb_m), make_shape(tb_k, k / tb_k));
    auto a_blk1_stride = make_stride(make_stride(tb_k, tb_m * k), make_stride(1, tb_m * tb_k));
    Layout a_blk1 = make_layout(a_blk1_shape, a_blk1_stride);
    print2D(a_blk1);
    print("\n");

    // Layout a_blk2 = make_layout(
    //     make_shape(
    //         make_shape(warp_m, tb_m / warp_m, m / tb_m),
    //         make_shape(warp_k, tb_k / warp_k, k / tb_k)
    //     ),
    //     make_stride(
    //         make_stride(warp_k, warp_m * tb_k, tb_m * k),
    //         make_stride(1, warp_m * warp_k, tb_m * tb_k)
    //     )
    // );
    // print2D(a_blk2);

    exit(0);
}

void pm(int *a) {
    for(int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%3d  ", a[i*4 + j]);
        }
        printf("\n");
    }
}

int main() {
    std::cout << "Blocked GEMM end-to-end example\n";
    std::cout << "M=" << M << " K=" << K << " N=" << N << "\n";
    std::cout << "TB_M=" << TB_M << " TB_N=" << TB_N << " TB_K=" << TB_K << "\n";

    // float a[M][K] = {
    //     {1, 2, 3, 4},
    //     {5, 6, 7, 8},
    //     {9,10,11,12},
    //     {13,14,15,16}
    // };
    // float aa[4][4];

    // block_A_host((float*)a, (float*)aa);
    
    // auto l = make_layout(
    //     make_shape(BM, BK, TB_M, TB_K),
    //     make_stride(K*TB_M, TB_M*TB_K, TB_K, 1)
    // );
    // auto t = make_tensor((float*)aa, l);

    // for(int bm = 0; bm < BM; bm++) {
    //     for(int bk = 0; bk < BK; bk++) {
    //         printf("Block A bm=%d bk=%d\n", bm, bk);
    //         auto blk = t(bm, bk, _, _);
    //         for(int tm = 0; tm < TB_M; tm++) {
    //             for(int tk = 0; tk < TB_K; tk++) {
    //                 printf("%3d  ", (int) blk(tm, tk));
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }

    // return 0;

    // example_and_quit();

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

    cute_blocked_gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K);

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
        cute_blocked_gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K);
    }
    cudaDeviceSynchronize();

    // -----------------------------
    // Benchmark (10 runs)
    // -----------------------------
    float total_ms = 0.f, min_ms = 1e9, max_ms = 0.f;

    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);

        cute_blocked_gemm_kernel<<<grid, block, smem_bytes>>>(dAblk, dBblk, dC, M, N, K);
        
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
    free(hAblk); free(hBblk);
    cudaFree(dAblk); cudaFree(dBblk); cudaFree(dC);

    return 0;
}


