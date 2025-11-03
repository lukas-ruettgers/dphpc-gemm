// src/core/plan.hpp
#pragma once
#include <cstdint>
#include <string>
#include <optional>
#include <stdexcept>
#include <cuda_runtime.h>
#include "problem.hpp"

namespace dphpc {
// ------------------------------ Plan (shared) --------------------------------
enum class BackendKind : uint8_t { WMMA, Cutlass, CuTe };

// ------------------------------ Plan (shared) --------------------------------
// NOTE: Plan is a runtime container of *choices*. It is intentionally
// non-templated and uses enums/ints so the planner/dispatcher can pass it
// around easily. Backend adaptors will translate these into concrete types.
struct Plan {
    virtual ~Plan() = default;          // <-- make base polymorphic
    
    BackendKind backend{BackendKind::CuTe};

    // Shared datatypes for kernel operands / scalars
    // (duplicated from Problem intentionally, so a planner may override)
    DataType typeA{DataType::f32};
    DataType typeB{DataType::f32};
    DataType typeC{DataType::f32};
    DataType typeD{DataType::f32};     // output / accumulator target
    DataType typeAlpha{DataType::f32};
    DataType typeBeta{DataType::f32};

    // Set scale values via cli (optional; set by caller if known at plan time)
    std::optional<double> alpha_value{};
    std::optional<double> beta_value{};

    // Common tiling knobs across backends
    int tile_m{128};
    int tile_n{128};
    int tile_k{8};
    int stages{2};             // pipeline stages (smem “k-tiles”)
    int split_k_slices{1};     // parallelism along K

    // Launch shape (the adaptor may override based on chosen layout/tiler)
    int block_threads{-1};     // -1 => adaptor derives from layout
    int grid_m{-1};            // -1 => adaptor derives from M/tile_m
    int grid_n{-1};            // -1 => adaptor derives from N/tile_n

    // Hints
    bool use_tensor_cores{true};
};


} // namespace dphpc