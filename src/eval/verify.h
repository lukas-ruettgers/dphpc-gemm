// src/eval/verify.h
#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#include "../core/problem.hpp"
#include "../core/plan.hpp"
#include "../core/dispatcher.hpp"
#include "../utils/cuda_check.hpp"
#include "../utils/data.hpp"

namespace dphpc::eval {

// Column-major index helper: elem(i,j) at i + j*ld
inline std::size_t idx_cm(int i, int j, int ld) {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// CPU reference: C_ref = alpha * op(A) * op(B) + beta * C0
inline void gemm_cpu_ref_f32(float*       C_ref, int ldC,
                             const float* A,     int ldA, bool transA,
                             const float* B,     int ldB, bool transB,
                             const float* C0,    int M, int N, int K,
                             float alpha, float beta)
{
    // Start from beta*C0
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float c = (beta == 0.0f) ? 0.0f : beta * C0[idx_cm(i, j, ldC)];
            C_ref[idx_cm(i, j, ldC)] = c;
        }
    }

    // Accumulate alpha * A_op * B_op
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            // B_op(k,j)
            float bkj;
            if (!transB) { // B is KxN, column-major
                bkj = B[idx_cm(k, j, ldB)];
            } else {       // B^T is KxN → B is NxK in storage
                bkj = B[idx_cm(j, k, ldB)];
            }
            float abcol_scale = alpha * bkj;

            for (int i = 0; i < M; ++i) {
                // A_op(i,k)
                float aik;
                if (!transA) { // A is MxK
                    aik = A[idx_cm(i, k, ldA)];
                } else {       // A^T is MxK → A is KxM in storage
                    aik = A[idx_cm(k, i, ldA)];
                }
                C_ref[idx_cm(i, j, ldC)] += aik * abcol_scale;
            }
        }
    }
}

inline bool within_tol(float got, float ref, double rtol, double atol) {
    double diff = std::fabs(static_cast<double>(got) - static_cast<double>(ref));
    double tol  = atol + rtol * std::fabs(static_cast<double>(ref));
    return diff <= tol;
}

inline bool verify_gemm(const Problem& problem,
                        const Plan& plan,
                        GemmEntryFn entry,
                        double rtol,
                        double atol,
                        cudaStream_t stream)
{
    // Currently support f32 end-to-end; extend when other dtypes are wired.
    if (problem.typeA != DataType::f32 ||
        problem.typeB != DataType::f32 ||
        problem.typeC != DataType::f32 ||
        problem.typeD != DataType::f32) {
        std::cerr << "verify_gemm: Only f32 is supported in verification at this stage.\n";
        return false;
    }

    const int M = static_cast<int>(problem.M);
    const int N = static_cast<int>(problem.N);
    const int K = static_cast<int>(problem.K);

    // Leading dimensions consistent with your CuTe baseline (column-major)
    int ldA = problem.transA ? K : M;
    int ldB = problem.transB ? N : K;
    int ldC = M;

    // Generate host data (reproducible, distinct streams)
    using dphpc::data::DistributionType;
    const std::uint64_t base_seed = GLOBAL_SEED;
    std::vector<float> hA, hB, hC0;

    // A buffer size depends on transpose (column-major physical dims)
    const int HA = problem.transA ? K : M;
    const int WA = problem.transA ? M : K;
    const int HB = problem.transB ? N : K;
    const int WB = problem.transB ? K : N;

    hA  = dphpc::data::generateRandomMatrix<float>(HA, WA, DistributionType::Gaussian,
            dphpc::data::mixed_seed(base_seed, 0), 0.0f, 1.0f);
    hB  = dphpc::data::generateRandomMatrix<float>(HB, WB, DistributionType::Gaussian,
            dphpc::data::mixed_seed(base_seed, 1), 0.0f, 1.0f);
    hC0 = dphpc::data::generateRandomMatrix<float>(M,  N,  DistributionType::Gaussian,
            dphpc::data::mixed_seed(base_seed, 2), 0.0f, 1.0f);

    // Copy of C for output
    std::vector<float> hC = hC0;         // will be overwritten by device result
    std::vector<float> hC_ref(M * N, 0); // CPU reference

    // Scalars
    const double alpha_d = plan.alpha_value.value_or(1.0);
    const double beta_d  = plan.beta_value.value_or(0.0);
    const float  alpha   = static_cast<float>(alpha_d);
    const float  beta    = static_cast<float>(beta_d);

    // Compute CPU reference
    gemm_cpu_ref_f32(hC_ref.data(), ldC,
                     hA.data(), ldA, problem.transA,
                     hB.data(), ldB, problem.transB,
                     hC0.data(), M, N, K,
                     alpha, beta);

    // Device buffers
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    const std::size_t bytesA = static_cast<std::size_t>(HA) * static_cast<std::size_t>(WA) * sizeof(float);
    const std::size_t bytesB = static_cast<std::size_t>(HB) * static_cast<std::size_t>(WB) * sizeof(float);
    const std::size_t bytesC = static_cast<std::size_t>(M)  * static_cast<std::size_t>(N)  * sizeof(float);

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(),  bytesA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dB, hB.data(),  bytesB, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dC, hC.data(),  bytesC, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Invoke backend entry (overwrites dC in place)
    CUDA_CHECK(entry(problem, plan,
                           dA, ldA,
                           dB, ldB,
                           dC, ldC,
                           alpha_d, beta_d,
                           stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Compare
    int mismatches = 0;
    const int kMaxReport = 10;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            const float got = hC[idx_cm(i, j, ldC)];
            const float ref = hC_ref[idx_cm(i, j, ldC)];
            if (!within_tol(got, ref, rtol, atol)) {
                if (mismatches < kMaxReport) {
                    const double diff = std::fabs(static_cast<double>(got) - static_cast<double>(ref));
                    const double rerr = diff / (std::fabs(static_cast<double>(ref)) + 1e-30);
                    std::cerr << "Mismatch at (i=" << i << ", j=" << j
                              << ") got=" << got << " ref=" << ref
                              << " abs_err=" << diff << " rel_err=" << rerr << "\n";
                }
                ++mismatches;
            }
        }
    }

    // Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    if (mismatches) {
        std::cerr << "verify_gemm: FAILED with " << mismatches << " mismatches out of "
                  << static_cast<long long>(M) * static_cast<long long>(N) << " elements.\n";
        return false;
    }

    std::cout << "verify_gemm: PASS (" << M << "x" << N << "x" << K
              << "), alpha=" << alpha_d << ", beta=" << beta_d << "\n";
    return true;
}

} // namespace dphpc::eval
