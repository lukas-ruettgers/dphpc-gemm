// src/core/dispatch.cpp 
#include "dispatcher.hpp"

// Backends: include only the headers that provide gemm_entry(...)
#include "../backend/cute/backend_adaptor.hpp"

// Eval (signatures only; implementations come later)
#include "../eval/bench.h"
#include "../eval/verify.h"

#include <stdexcept>

namespace dphpc {

GemmEntryFn get_gemm_entry(const Plan& plan) {
    switch (plan.backend) {
        case BackendKind::CuTe:
            return &backend::cute::gemm_entry;
        case BackendKind::Cutlass:
        case BackendKind::WMMA:
            // Add when those backends are implemented
            throw std::runtime_error("Dispatcher: backend not implemented yet.");
        default:
            throw std::runtime_error("Dispatcher: unknown backend.");
    }
}

int dispatch_and_run(const Problem& problem,
                     const Plan& plan,
                     bool verify,
                     cudaStream_t stream)
{
    GemmEntryFn entry = get_gemm_entry(plan);

    if (verify) {
        // Simple default tolerances for FP32; adjust per dtype later
        const double rtol = 1e-4, atol = 1e-6;
        const bool ok = eval::verify_gemm(problem, plan, entry, rtol, atol, stream);
        if (!ok) throw std::runtime_error("Verification failed.");
        return 0;
    }

    // Benchmark defaults; tune later or pass via CLI
    const int warmup = 5;
    const int iters  = 50;
    (void)eval::bench_gemm(problem, plan, entry, warmup, iters, stream);
    return 0;
}

} // namespace dphpc



/**
  cute::device_init(0);

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  double gflops = (2.0*m*n*k) * 1e-9;

  const int timing_iterations = 100;
  GPU_Clock timer;

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }
  // Run once
  d_C = h_C;
  gemm(transA, transB, m, n, k,
       alpha,
       d_A.data().get(), ldA,
       d_B.data().get(), ldB,
       beta,
       d_C.data().get(), ldC);
  CUTE_CHECK_LAST();
  thrust::host_vector<TC> cute_result = d_C;

  // Timing iterations
  timer.start();
  for (int i = 0; i < timing_iterations; ++i) {
    gemm(transA, transB, m, n, k,
         alpha,
         d_A.data().get(), ldA,
         d_B.data().get(), ldB,
         beta,
         d_C.data().get(), ldC);
  }
  double cute_time = timer.seconds() / timing_iterations;
  CUTE_CHECK_LAST();
  printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);
  return 0;
 */