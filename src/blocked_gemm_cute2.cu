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

constexpr int W_M = 8;
constexpr int W_N = 4;
constexpr int W_K = 4;

// Tile sizes (tune as desired)
constexpr int TB_M = 32;
constexpr int TB_N = 32;
constexpr int TB_K = 32;

// Matrix sizes (must be multiples of TB sizes in this simple demo)
constexpr int M = 1024;   // rows of A and C
constexpr int N = 1024;   // cols of B and C
constexpr int K = 1024;   // cols of A, rows of B

constexpr int WpTB_M = TB_M / W_M;
constexpr int WpTB_N = TB_N / W_N;
constexpr int WpTB_K = TB_K / W_K;

// Derived block counts
constexpr int BM = M / TB_M;
constexpr int BN = N / TB_N;
constexpr int BK = K / TB_K;

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
void block_host_matrix(
    const float* mat, // [in] row-major A x B
    float* matblk, // [out] contiguous blocked: [BA][BB][WpTB_A][WpTB_B][W_A][W_B]
    int A, int B,
    int BA, int BB,
    int TB_A, int TB_B,
    int WpTB_A, int WpTB_B,
    int W_A, int W_B
)
{
    for (int ba = 0; ba < BA; ++ba) {
        for (int bb = 0; bb < BB; ++bb) {
            // pointer to destination block
            size_t block_index = (size_t)ba * BB + bb;
            float* dst_block = matblk + block_index * (TB_A * TB_B);
            // printf("Block (row %d, col %d): index %lu\n", ba, bb, block_index);

            for (int wa = 0; wa < WpTB_A; wa++) {
                for (int wb = 0; wb < WpTB_B; wb++) {
                    size_t warp_index = (size_t) wa * WpTB_B + wb;
                    float *dst_warp = dst_block + warp_index * (W_A * W_B);

                    // printf("  Warp (row %d, col %d): index %lu\n", wa, wb, warp_index);

                    for (int i = 0; i < W_A; ++i) {
                        for (int j = 0; j < W_B; ++j) {
                            int global_a = ba * TB_A + wa * W_A + i;
                            int global_b = bb * TB_B + wb * W_B + j;

                            // printf("    Element (row %d, col %d): global indices (%d, %d)\n", i, j, global_m, global_k);

                            dst_warp[i * W_B + j] = mat[global_a * B + global_b];
                        }
                    }
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

    const int tid = threadIdx.x;
    const int warpid = tid / 32;
    const int warp_tid = tid % 32;

    // Warp identification - each warp computes one warp block in WpTB_M x WpTB_N tile.
    const int wm = warpid / WpTB_N;
    const int wn = warpid % WpTB_N;

    if (wm >= WpTB_M || wn >= WpTB_N) return;

    // Thread identification - each thread computes one element in TB_M x TB_N tile
    const int tm = warp_tid / W_N; // 0..W_M-1
    const int tn = warp_tid % W_N; // 0..W_N-1

    if (tm >= W_M || tn >= W_N) return;

    // Shared memory allocation
    // Shared memory for entire TB. But each warp will only use it's own section.
    extern __shared__ float smem[]; // size = TB_M*TB_K + TB_K*TB_N
    float* smem_A = smem;                       // TB_M * TB_K
    float* smem_B = smem + TB_M * TB_K;        // TB_K * TB_N


    // Create CUTE tensors for shared memory
    Layout sA_layout = make_layout(
        make_shape(WpTB_M, WpTB_K, W_M, W_K),
        make_stride(TB_K * W_M, W_M * W_K, W_K, 1)
    );
    Layout sB_layout = make_layout(
        make_shape(WpTB_K, WpTB_N, W_K, W_N),
        make_stride(TB_N * W_K, W_K * W_N, W_N, 1)
    );

    Tensor sA = make_tensor(make_smem_ptr(smem_A), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem_B), sB_layout);

    auto Ablk_layout = make_layout(
        make_shape(BM, BK, WpTB_M, WpTB_K, W_M, W_K),
        make_stride(K * TB_M, TB_M * TB_K, TB_K * W_M, W_M * W_K, W_K, 1)
    );
    auto Bblk_layout = make_layout(
        make_shape(BK, BN, WpTB_K, WpTB_N, W_K, W_N),
        make_stride(N * TB_K, TB_K * TB_N, TB_N * W_K, W_K * W_N, W_N, 1)
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

    // Loop over bk blocks and wk warps
    for (int bk = 0; bk < BK; ++bk) {
        for (int wk = 0; wk < WpTB_K; ++wk) {
            // Extract the current warp using CUTE slicing
            // Get A block: gAblk[bm, bk, :, :] - a TB_M x TB_K tile
            auto gA_warp = gAblk(bm, bk, wm, wk, _, _);
            // Get B block: gBblk[bk, bn, :, :] - a TB_K x TB_N tile
            auto gB_warp = gBblk(bk, bn, wk, wn, _, _);

            // Slice the shared memory for this warp.
            auto sA_warp = sA(wm, wk, _, _);
            auto sB_warp = sB(wk, wn, _, _);

            // Load A tile
            for (int idx = warp_tid; idx < W_M * W_K; idx += W_M * W_N) {
                int i = idx / W_K;
                int j = idx % W_K;
                sA_warp(i, j) = gA_warp(i, j);
            }

            // Load B tile  
            for (int idx = warp_tid; idx < W_K * W_N; idx += W_M * W_N) {
                int i = idx / W_N;
                int j = idx % W_N;
                sB_warp(i, j) = gB_warp(i, j);
            }

            __syncwarp();

            // Compute partial product using CUTE tensor access
            // This thread computes: accum += sum_over_kk(sA[tm, kk] * sB[kk, tn])
            for (int kk = 0; kk < W_K; ++kk) {
                accum += sA_warp(tm, kk) * sB_warp(kk, tn);
            }

            __syncwarp();
        }
    }

    // Write result using CUTE tensor access
    int row = bm * TB_M + wm * W_M + tm;
    int col = bn * TB_N + wn * W_N + tn;
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
    //     {1, 2, 3, 4, 5, 6, 7, 8},
    //     {9,10,11,12,13,14,15,16},
    //     {17,18,19,20,21,22,23,24},
    //     {25,26,27,28,29,30,31,32},
    //     {33,34,35,36,37,38,39,40},
    //     {41,42,43,44,45,46,47,48},
    //     {49,50,51,52,53,54,55,56},
    //     {57,58,59,60,61,62,63,64}
    // };
    // float aa[M][K];

    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         printf("%3d  ", (int) a[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // block_A_host((float*)a, (float*)aa);
    
    // auto l = make_layout(
    //     make_shape(BM, BK, WpTB_M, WpTB_K, W_M, W_K),
    //     make_stride(K * TB_M, TB_M * TB_K, TB_K * W_M, W_M * W_K, W_K, 1)
    // );
    // auto t = make_tensor((float*)aa, l);

    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         printf("%3d  ", (int) aa[i][j]);
    //     }
    //     printf("\n");
    // }

    // for(int bm = 0; bm < BM; bm++) {
    //     for(int bk = 0; bk < BK; bk++) {
    //         printf("Block A bm=%d bk=%d\n", bm, bk);
    //         auto blk = t(bm, bk, _, _, _, _);
            
    //         for(int wm = 0; wm < WpTB_M; wm++) {
    //             for(int wk = 0; wk < WpTB_K; wk++) {
    //                 printf("Warp A wm=%d wk=%d\n", wm, wk);
    //                 auto warp = blk(wm, wk, _, _);

    //                 for(int tm = 0; tm < W_M; tm++) {
    //                     for(int tk = 0; tk < W_K; tk++) {
    //                         printf("%3d  ", (int) warp(tm, tk));
    //                     }
    //                     printf("\n");
    //                 }
    //                 printf("\n");
    //             }
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
    block_host_matrix(hA, hAblk, M, K, BM, BK, TB_M, TB_K, WpTB_M, WpTB_K, W_M, W_K);
    block_host_matrix(hB, hBblk, K, N, BK, BN, TB_K, TB_N, WpTB_K, WpTB_N, W_K, W_N);

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

    return 0;


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


