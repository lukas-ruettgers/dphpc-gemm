// src/core/dispatcher.hpp
#pragma once
#include <cuda_runtime.h>
#include "problem.hpp"
#include "plan.hpp"

namespace dphpc {

// ---------------------------------------------------------------------------
// Unified entrypoint type for GEMM backends
// ---------------------------------------------------------------------------
//
// Every backend adaptor must implement a function matching this signature:
//
//   cudaError_t gemm_entry(const Problem& problem,
//                          const Plan& plan,
//                          const void* A, int ldA,
//                          const void* B, int ldB,
//                          void* C, int ldC,
//                          double alpha, double beta,
//                          cudaStream_t stream);
//
// The backend implementation may static_cast/dynamic_cast plan to its
// concrete type (e.g., PlanCuTe) before launching kernels.
//
using GemmEntryFn = cudaError_t(*)(const Problem&,
                                   const Plan&,
                                   const void* A, int ldA,
                                   const void* B, int ldB,
                                   void*       C, int ldC,
                                   double alpha, double beta,
                                   cudaStream_t stream);

// Return the backend’s entry function based on plan.backend.
GemmEntryFn get_gemm_entry(const Plan& plan);

// Invoke either benchmark or verification, depending on `verify`.
// Returns 0 on success (for now).
int dispatch_and_run(const Problem& problem,
                     const Plan& plan,
                     bool verify,
                     cudaStream_t stream = 0);

} // namespace dphpc
