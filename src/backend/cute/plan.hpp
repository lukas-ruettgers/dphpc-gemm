// src/backend/cute/plan.hpp
#pragma once
#include <cstdint>
#include <string>
#include "../../core/plan.hpp"

namespace dphpc::backend::cute {

// Simple tags for choosing layouts at runtime.
// The adaptor maps these to CuTe types (e.g., default vs LayoutRight).
enum class SmemLayoutTag : uint8_t {
    RowMajor,   // e.g., make_layout(make_shape(bM, bK)) for A; (bN,bK) for B; (bM,bN) for C
    ColMajor,   // e.g., make_layout(..., LayoutRight{}) as needed
    KMajor      // explicit “k-major” intent (mapped to LayoutRight for (m,k)/(n,k))
};

enum class ThreadLayoutTag : uint8_t {
    RowMajor,   // default thread mapping
    ColMajor,   // right-major
    KMajor
};

// ------------------------------ PlanCuTe -------------------------------------
// This extends Plan with CuTe-specific knobs that correspond to the tutorial
// baseline kernel parameters. Values are runtime choices; the adaptor turns
// them into the actual CuTe template/value arguments.
struct PlanCuTe : public dphpc::Plan {
    // Choose which transpose shape path (NT vs TN) to use in the adaptor
    // (The adaptor can also read from Problem::transA/transB, but this lets
    //  the planner pin the exact variant explicitly.)
    enum class Path : uint8_t { NT, TN };
    Path path{Path::NT};

    // CTA tile (static in tutorials, but we keep ints for planning flexibility)
    int blk_m{128};
    int blk_n{128};
    int blk_k{8};

    // Shared-memory layouts (A/B/C) for the chosen path
    SmemLayoutTag sA{SmemLayoutTag::RowMajor}; // A smem layout tag
    SmemLayoutTag sB{SmemLayoutTag::ColMajor}; // B smem layout tag
    SmemLayoutTag sC{SmemLayoutTag::RowMajor}; // C smem layout tag

    // Thread layouts (A/B/C)
    ThreadLayoutTag tA{ThreadLayoutTag::RowMajor};
    ThreadLayoutTag tB{ThreadLayoutTag::RowMajor};
    ThreadLayoutTag tC{ThreadLayoutTag::RowMajor};

    // Optional: override derived launch if desired
    int override_block_threads{-1};  // -1 => derive from tC
    int override_grid_m{-1};
    int override_grid_n{-1};

    // Convenience ctor sets backend kind and propagates shared tiles
    PlanCuTe() {
        backend = dphpc::BackendKind::CuTe;
        tile_m = blk_m; tile_n = blk_n; tile_k = blk_k;
    }
};

} // namespace dphpc::backend::cute
