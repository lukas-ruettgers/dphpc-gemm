#include "../../core/plan.hpp"
#include <cstdint>
#include <string>
#include <optional>
#include <stdexcept>
#include <cuda_runtime.h>

namespace dphpc {
// --------------------------- Backend: WMMA -----------------------------------
struct PlanWMMA : public Plan {
    // Example WMMA specifics (extend as needed)
    // fragment sizes, MMA shape, layouts, etc.
    int mma_m{16}, mma_n{16}, mma_k{16};  // wmma::matrix_a/b accumulator tile
    bool row_major_a{true};
    bool row_major_b{false};
    bool row_major_c{true};

    BackendKind backend() const noexcept override { return BackendKind::WMMA; }
};


// ---------------------- Baseline launcher stubs ------------------------------
// These are *demonstration* entry points. Replace the bodies with calls into
// your actual backend code (e.g., CUTLASS/CuTe tutorial kernels). The function
// signatures are stable so planner/dispatcher can call them uniformly.

/** WMMA baseline */
inline cudaError_t run_baseline(const Problem& p, const PlanWMMA& plan,
                                const void* A, const void* B, const void* C, void* D,
                                double alpha, double beta,
                                cudaStream_t stream = 0)
{
    // TODO: launch your WMMA GEMM kernel here.
    //      (block/grid dims from plan, tiles from plan, etc.)
    // Example shape checks (minimal):
    (void)A; (void)B; (void)C; (void)D; (void)alpha; (void)beta; (void)stream;
    if (p.M<=0 || p.N<=0 || p.K<=0) return cudaErrorInvalidValue;
    return cudaSuccess;
}

} // namespace dphpc
