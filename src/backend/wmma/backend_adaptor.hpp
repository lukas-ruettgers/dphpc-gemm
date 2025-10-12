#pragma once
#include "../../core/dispatcher.hpp"

namespace dphpc::backend::wmma {
cudaError_t gemm_entry(const Problem& problem,
                       const Plan& plan,
                       const void* A, int ldA,
                       const void* B, int ldB,
                       void* C, int ldC,
                       double alpha, double beta,
                       cudaStream_t stream = 0);
}
