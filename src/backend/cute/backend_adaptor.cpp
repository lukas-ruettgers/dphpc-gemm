// src/backend/cute/backend_adaptor.cpp
#include "backend_adaptor.hpp"           // decl: gemm_entry(...)
#include "../../core/dispatcher.hpp"     // GemmEntryFn type (for consistency)
#include "../../core/problem.hpp"
#include "../../core/plan.hpp"
#include "plan.hpp"                      // dphpc::backend::cute::PlanCuTe

#include <cuda_runtime.h>
#include <stdexcept>

namespace dphpc::backend::cute {

// -----------------------------------------------------------------------------
// Forward declaration of the host launcher implemented in gemm.cu
// This function will read PlanCuTe (tiles/layout tags), build CuTe shapes,
// and launch gemm_device<<<...>>>(...).
// -----------------------------------------------------------------------------
void gemm(const PlanCuTe& plan,
          char transA, char transB,
          int m, int n, int k,
          float alpha,
          const float* A, int ldA,
          const float* B, int ldB,
          float  beta,
          float* C, int ldC,
          cudaStream_t stream);

// -----------------------------------------------------------------------------
// Unified entrypoint used by dispatcher/bench/verify
// -----------------------------------------------------------------------------
cudaError_t gemm_entry(const Problem& problem,
                       const Plan& plan,
                       const void* A, int ldA,
                       const void* B, int ldB,
                       void*       C, int ldC,
                       double alpha, double beta,
                       cudaStream_t stream)
{
    // Ensure correct backend
    if (plan.backend != BackendKind::CuTe) {
        return cudaErrorInvalidDeviceFunction;
    }

    // Downcast to PlanCuTe (planner guarantees this when backend==CuTe)
    const auto* p = dynamic_cast<const PlanCuTe*>(&plan);
    if (!p) {
        return cudaErrorInvalidValue;
    }

    // Currently support f32 end-to-end (extend with dtype switch later)
    if (problem.typeA != DataType::f32 ||
        problem.typeB != DataType::f32 ||
        problem.typeC != DataType::f32 ||
        problem.typeD != DataType::f32) {
        return cudaErrorNotSupported;
    }

    // Shapes and transposition flags
    const int m = static_cast<int>(problem.M);
    const int n = static_cast<int>(problem.N);
    const int k = static_cast<int>(problem.K);
    const char ta = problem.transA ? 'T' : 'N';
    const char tb = problem.transB ? 'T' : 'N';

    // Cast raw pointers based on dtype (f32 path only for now)
    const float* Af = static_cast<const float*>(A);
    const float* Bf = static_cast<const float*>(B);
    float*       Cf = static_cast<float*>(C);

    // Forward to the CuTe host launcher defined in gemm.cu
    gemm(*p, ta, tb, m, n, k,
         static_cast<float>(alpha),
         Af, ldA,
         Bf, ldB,
         static_cast<float>(beta),
         Cf, ldC,
         stream);

    // Non-throwing contract: return the async kernel status
    return cudaGetLastError();
}

} // namespace dphpc::backend::cute
