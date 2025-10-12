#pragma once
#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>

#include "../core/problem.hpp"
#include "../core/plan.hpp"
#include "../core/dispatcher.hpp"
#include "../utils/cuda_check.hpp"
#include "../utils/timing.hpp"
#include "../utils/data.hpp"

namespace dphpc::eval {

// column-major index helper
inline std::size_t idx_cm(int i, int j, int ld) {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// Returns mean milliseconds (also prints mean/stddev GFLOP/s)
inline double bench_gemm(const Problem& problem,
                         const Plan& plan,
                         GemmEntryFn entry,
                         int warmup_iters,
                         int iters,
                         cudaStream_t stream = 0)
{
    // Current scope: f32 end-to-end (extend later with dtype switches)
    if (problem.typeA != DataType::f32 ||
        problem.typeB != DataType::f32 ||
        problem.typeC != DataType::f32 ||
        problem.typeD != DataType::f32) {
        std::cerr << "bench_gemm: Only f32 supported in this benchmark stage.\n";
        return 0.0;
    }

    const int M = static_cast<int>(problem.M);
    const int N = static_cast<int>(problem.N);
    const int K = static_cast<int>(problem.K);

    // Column-major leading dimensions consistent with your baseline
    int ldA = problem.transA ? K : M;
    int ldB = problem.transB ? N : K;
    int ldC = M;

    // Physical (stored) matrix sizes given transposes (column-major)
    const int HA = problem.transA ? K : M;
    const int WA = problem.transA ? M : K;
    const int HB = problem.transB ? N : K;
    const int WB = problem.transB ? K : N;

    const std::size_t bytesA = static_cast<std::size_t>(HA) * static_cast<std::size_t>(WA) * sizeof(float);
    const std::size_t bytesB = static_cast<std::size_t>(HB) * static_cast<std::size_t>(WB) * sizeof(float);
    const std::size_t bytesC = static_cast<std::size_t>(M)  * static_cast<std::size_t>(N)  * sizeof(float);

    // Scalars
    const double alpha_d = plan.alpha_value.value_or(1.0);
    const double beta_d  = plan.beta_value.value_or(0.0);

    // Host buffers (reused each iter; re-filled with fresh RNG)
    std::vector<float> hA(HA * WA), hB(HB * WB), hC(M * N);

    // Device buffers (persistent across iters)
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    const std::uint64_t base_seed = GLOBAL_SEED;

    auto reinit_and_copy = [&](int iter){
        // Different seeds per matrix & iteration (reproducible)
        using dphpc::data::DistributionType;
        auto sA = dphpc::data::mixed_seed(base_seed, (0ull << 32) ^ static_cast<std::uint64_t>(iter));
        auto sB = dphpc::data::mixed_seed(base_seed, (1ull << 32) ^ static_cast<std::uint64_t>(iter));
        auto sC = dphpc::data::mixed_seed(base_seed, (2ull << 32) ^ static_cast<std::uint64_t>(iter));

        dphpc::data::fillRandomMatrix<float>(hA.data(), HA, WA, DistributionType::Gaussian, sA, 0.0f, 1.0f);
        dphpc::data::fillRandomMatrix<float>(hB.data(), HB, WB, DistributionType::Gaussian, sB, 0.0f, 1.0f);
        dphpc::data::fillRandomMatrix<float>(hC.data(), M,  N,  DistributionType::Gaussian, sC, 0.0f, 1.0f);

        CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(), bytesA, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dB, hB.data(), bytesB, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dC, hC.data(), bytesC, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // Warmups (reinitialize each time, but do NOT time)
    for (int i = 0; i < warmup_iters; ++i) {
        reinit_and_copy(i);
        CUDA_CHECK(entry(problem, plan,
                               dA, ldA,
                               dB, ldB,
                               dC, ldC,
                               alpha_d, beta_d,
                               stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Timed iterations: reinit each time, but time ONLY the GEMM call via events
    std::vector<double> times_ms;
    times_ms.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        reinit_and_copy(warmup_iters + i);

        dphpc::timing::CudaTimer t(stream);
        t.record_start();
        CUDA_CHECK(entry(problem, plan,
                               dA, ldA,
                               dB, ldB,
                               dC, ldC,
                               alpha_d, beta_d,
                               stream));
        t.record_stop();
        // Ensure kernel completion for accurate elapsed time
        const double ms = static_cast<double>(t.elapsed_ms());
        times_ms.push_back(ms);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK_LAST();

    // Summarize & report
    auto [mean_ms, std_ms] = dphpc::timing::summarize(times_ms, 0.05);

    // FLOPs for GEMM: 2*M*N*K (mul+add); report GFLOP/s
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double gflops_mean = flops / (mean_ms * 1.0e6);
    const double gflops_std  = (std_ms > 0.0 ? flops / (1.0e6) * (1.0 / (mean_ms*mean_ms)) * std_ms : 0.0);

    std::cout << "bench_gemm: "
              << M << "x" << N << "x" << K
              << "  iters=" << iters
              << "  mean=" << mean_ms << " ms"
              << "  std="  << std_ms  << " ms"
              << "  GFLOP/s(mean)=" << gflops_mean
              << "  ~std≈" << gflops_std
              << "  alpha=" << alpha_d
              << "  beta="  << beta_d
              << (problem.transA ? "  transA=T" : "  transA=N")
              << (problem.transB ? "  transB=T" : "  transB=N")
              << "\n";

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    return mean_ms;
}

} // namespace dphpc::eval
