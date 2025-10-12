#include "plan.hpp"
#include "utils/cuda_check.hpp"

namespace dphpc {

inline cudaError_t dispatch(const Problem& p, const Plan& base,
                            const void* A, const void* B, const void* C, void* D,
                            double alpha, double beta, cudaStream_t stream = 0)
{
    switch (base.backend()) {
        case BackendKind::WMMA:
            return run_baseline(p, static_cast<const PlanWMMA&>(base), A,B,C,D, alpha, beta, stream);
        case BackendKind::Cutlass:
            return run_baseline(p, static_cast<const PlanCutlass&>(base), A,B,C,D, alpha, beta, stream);
        case BackendKind::CuTe:
            return run_baseline(p, static_cast<const PlanCuTe&>(base), A,B,C,D, alpha, beta, stream);
        default:
            return cudaErrorInvalidDeviceFunction;
    }
}

} // namespace dphpc
