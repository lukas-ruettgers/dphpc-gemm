// src/core/planner.hpp
#pragma once
#include <memory>
#include <stdexcept>

#include "problem.hpp"
#include "plan.hpp"

// include the backend-specific plan type (CuTe) so we can construct it
#include "backend/cute/plan.hpp"

namespace dphpc {

// Return a backend-specific Plan for the given Problem.
// For now: if CuTe is available, return a default PlanCuTe consistent with the
// example kernel setup (tiles 128x128x8, row-major smem/thread layouts).
inline std::unique_ptr<Plan> make_plan(const Problem& pr) {
    if (pr.cute_available) {
        // Return default CuTe for now
        auto p = std::make_unique<backend::cute::PlanCuTe>();

        // Backend kind is set in PlanCuTe ctor, but keep explicit for clarity
        p->backend = BackendKind::CuTe;

        // Copy shared/operand/scalar dtypes from Problem
        p->typeA     = pr.typeA;
        p->typeB     = pr.typeB;
        p->typeC     = pr.typeC;
        p->typeD     = pr.typeD;
        p->typeAlpha = pr.typeD;   // common choice: accumulate/scale in output type
        p->typeBeta  = pr.typeD;

        // Set shared tiles/stages consistent with your example
        p->blk_m = 128;
        p->blk_n = 128;
        p->blk_k = 8;
        p->tile_m = p->blk_m;
        p->tile_n = p->blk_n;
        p->tile_k = p->blk_k;
        p->stages = 2;             // tutorial-style baseline

        // Smem layouts and thread layouts (row-major tags, adaptor maps to CuTe)
        p->sA = backend::cute::SmemLayoutTag::RowMajor;  // m-major for A
        p->sB = backend::cute::SmemLayoutTag::RowMajor;  // n-major for B
        p->sC = backend::cute::SmemLayoutTag::RowMajor;  // m-major for C

        p->tA = backend::cute::ThreadLayoutTag::RowMajor; // 32x8 style (adaptor decides)
        p->tB = backend::cute::ThreadLayoutTag::RowMajor; // 32x8
        p->tC = backend::cute::ThreadLayoutTag::RowMajor; // 16x16

        // Launch shape left to adaptor unless explicitly overridden
        p->override_block_threads = -1;
        p->override_grid_m = -1;
        p->override_grid_n = -1;

        // Choose NT / TN path from transposes.
        // Supported now: A not transposed & B transposed  -> NT
        //                A transposed     & B not transposed -> TN
        if (!pr.transA &&  pr.transB) {
            p->path = backend::cute::PlanCuTe::Path::NT;
        } else if (pr.transA && !pr.transB) {
            p->path = backend::cute::PlanCuTe::Path::TN;
        } else {
            // Other combos (NN/TT) not implemented in baseline
            throw std::runtime_error(
                "Planner: only NT and TN paths are supported at this stage (got "
                + std::string(pr.transA ? "T" : "N") + std::string(pr.transB ? "T" : "N") + ").");
        }

        // Tensor core hint: enable by default when available
        p->use_tensor_cores = true;

        return p;
    }

    // If no backend is available (or supported), fail fast for now
    throw std::runtime_error("Planner: no available backend (CuTe required at this stage).");
}

} // namespace dphpc
