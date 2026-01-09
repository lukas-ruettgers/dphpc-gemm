// gemm_blocked_example.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>
#include <cassert>
#include <cooperative_groups.h>

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
constexpr int TB_M = 128;
constexpr int TB_N = 128;
constexpr int TB_K = 16;

// Matrix sizes (runtime)
int M = 1024;
int K = 1024;
int N = 1024;

constexpr int TM = 8;
constexpr int TN = 4;

constexpr int WARPSIZE = 32;
constexpr int WM = 64;
constexpr int WN = 64;
constexpr int WNITER= 4;
constexpr int NUM_THREADS = 128;

// Derived block counts (runtime)
int BM, BK, BN;



#include <cuda/barrier>

template <int TB_M_, int TB_N_, int TB_K_>
__global__ void __launch_bounds__(NUM_THREADS) gemm_kernel(
    const float* Ablk,
    const float* Bblk,
    float* __restrict__ C,
    int M_, int N_, int K_,
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

    const int thread_idx_in_warp = threadIdx.x % WARPSIZE;
    const int thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const int thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);


    // How many threads are there in one row of the tile?
    const int threads_per_row = TB_N_ / TN;
    const int tm = tid / threads_per_row; // 0..TB_M-1
    const int tn = tid % threads_per_row; // 0..TB_N-1

    if (tm >= TB_M_ || tn >= TB_N_) return;

    // // Shared memory for tiles
    // extern __shared__ float smem[]; // size = TB_M*TB_K + TB_K*TB_N
    // float* sA = smem;                       // TB_M * TB_K
    // float* sB = smem + TB_M_ * TB_K_;        // TB_K * TB_N

    extern __shared__ char smem_raw[];

    float* smem = reinterpret_cast<float*>(
        (reinterpret_cast<uintptr_t>(smem_raw) + 15) & ~uintptr_t(15)
    ); 

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__  cuda::barrier<cuda::thread_scope_block> bar;

    // 3. Initialize the barrier (Only one thread does this)
    if (tid == 0) {
        init(&bar, blockDim.x); 
    }
    __syncthreads(); // Ensure barrier is ready

    auto group = cooperative_groups::this_thread_block();
    // Partition the single smem blob into 4 parts:
    // sA_buf0, sA_buf1, sB_buf0, sB_buf1
    float* sA0 = smem;
    float* sA1 = sA0 + (TB_M_ * TB_K_);
    float* sB0 = sA1 + (TB_M_ * TB_K_);
    float* sB1 = sB0 + (TB_K_ * TB_N_);

    // To index them easily like sA[read_buffer]:
    float* sA[2] = {sA0, sA1};
    float* sB[2] = {sB0, sB1};


    // Allocate thread-local cache for results in registerfile
    float accum[WMITER * TM * WNITER * TN] = {0.0f};
    float register_m[WMITER * TM] = {0.0f};
    float register_n[WNITER * TN] = {0.0f};

    int read_buffer = 0;
    int write_buffer = 0;
    //Load the first tile into buffer 0

    cuda::barrier<cuda::thread_scope_block>::arrival_token token;
    // --- PROLOGUE: Start loading the first tile (bk = 0) ---
    {
        const float* src_A = &Ablk[bm * TB_M_];
        #pragma unroll
        for (int k = 0; k < TB_K_; ++k, src_A += M_) {
            // A source: row (0 * TB_K + k), col (bm * TB_M)
            cuda::memcpy_async(group, &sA[0][k * TB_M_], src_A, cuda::aligned_size_t<16>(sizeof(float) * TB_M_), bar);
        }

        const float* src_B = &Bblk[bn * TB_N_];
        #pragma unroll
        for (int k = 0; k < TB_K_; ++k, src_B += N_) {
            // B source: row (0 * TB_K + k), col (bn * TB_N)
            cuda::memcpy_async(group, &sB[0][k * TB_N_], src_B, cuda::aligned_size_t<16>(sizeof(float) * TB_N_), bar);
        }
        bar.arrive_and_wait();
    }

    // Loop over bk blocks
    for (int bk = 0; bk < BK_; ++bk) {

        // Start loading the NEXT tile (bk + 1) while we compute the current one
        write_buffer = read_buffer ^ 1;
        if (bk + 1 < BK_) {
            const float* src_A = &Ablk[((bk + 1) * TB_K_) * M_ + (bm * TB_M_)];
            #pragma unroll
            for (int k = 0; k < TB_K_; ++k, src_A += M_) {
                // Next A: row ((bk+1) * TB_K + k), col (bm * TB_M)
                cuda::memcpy_async(group, &sA[write_buffer][k * TB_M_], src_A, cuda::aligned_size_t<16>(sizeof(float) * TB_M_), bar);
            }

            const float* src_B = &Bblk[((bk + 1) * TB_K_) * N_ + (bn * TB_N_)];
            #pragma unroll
            for (int k = 0; k < TB_K_; ++k, src_B += N_) {
                // Next B: row ((bk+1) * TB_K + k), col (bn * TB_N)
                cuda::memcpy_async(group, &sB[write_buffer][k * TB_N_], src_B, cuda::aligned_size_t<16>(sizeof(float) * TB_N_), bar);
            }
            
            token = bar.arrive(); 
        }
        
        // compute partial product for this thread's (tm,tn) from read buffer
        #pragma unroll
        for (int kk = 0; kk < TB_K_; ++kk) {
            #pragma unroll
            for (int wsub_row_idx = 0; wsub_row_idx < WMITER; ++wsub_row_idx){
                // #pragma unroll
                for (int i = 0;i < TM;  i++){
                    register_m[wsub_row_idx * TM + i] =  sA[read_buffer][(kk * TB_M_) + warp_row * WM + wsub_row_idx * WSUBM +
                           thread_row_in_warp * TM + i];
                }
            }
            #pragma unroll
            for (int wsub_col_idx = 0; wsub_col_idx < WNITER; ++wsub_col_idx)
            {
                // #pragma unroll
                for (int i = 0;i < TN; i++){
                    register_n[wsub_col_idx * TN + i] = sB[read_buffer][(kk * TB_N_) + warp_col * WN + wsub_col_idx * WSUBN +
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
        // Before starting the next iteration, wait for the async load to finish
        if (bk + 1 < BK_) {
            bar.wait(std::move(token));
        }
        read_buffer = write_buffer;
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
    if (argc == 4) {
        M    = std::atoi(argv[1]);
        N    = std::atoi(argv[2]);
        K    = std::atoi(argv[3]);
    } else {
        std::cout << "Usage: ./gemm M N K\n";
        std::cout << "Using default values.\n";
    }

    // Derived sizes
    BM = M / TB_M;
    BN = N / TB_N;
    BK = K / TB_K;

    std::cout << "\n\nBlocked GEMM (double buf async)\n";
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
    size_t smem_bytes = 2 * (TB_M * TB_K + TB_K * TB_N) * sizeof(float);

    std::cout << "Launching kernel grid(" << BM << "," << BN << ") block(" << NUM_THREADS << ") smem=" << smem_bytes << "\n";

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
    free(hA); free(hB); free(hC); free(hC_ref); free(hA_t);
    cudaFree(dAblk); cudaFree(dBblk); cudaFree(dC);

    return 0;
}


