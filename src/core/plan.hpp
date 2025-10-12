// src/core/plan.hpp
#pragma once
#include <cstdint>
#include <string>
#include <optional>
#include <stdexcept>
#include <cuda_runtime.h>

namespace dphpc {

// ----------------------------- Problem --------------------------------------
enum class DataType : uint8_t {
    f16, bf16, tf32, f32, f64, i8, i32
};

struct Problem {
    // Shapes
    int64_t M{0}, N{0}, K{0};

    // Transposition flags (true == transposed)
    bool transA{false};
    bool transB{false};
    bool transC{false};

    // Datatypes
    DataType typeA{DataType::f32};
    DataType typeB{DataType::f32};
    DataType typeC{DataType::f32};  // input C
    DataType typeD{DataType::f32};  // output D/accum target

    // Backend availability flags
    bool has_wmma{true};
    bool has_cutlass{true};
    bool has_cute{true};

    // Convenience
    std::string repr() const {
        auto dt2s = [](DataType t)->const char*{
            switch (t) {
                case DataType::f16: return "f16";
                case DataType::bf16: return "bf16";
                case DataType::tf32: return "tf32";
                case DataType::f32: return "f32";
                case DataType::f64: return "f64";
                case DataType::i8:  return "i8";
                case DataType::i32: return "i32";
            }
            return "?";
        };
        return "Problem{ M=" + std::to_string(M) +
               " N=" + std::to_string(N) +
               " K=" + std::to_string(K) +
               " tA=" + std::to_string(transA) +
               " tB=" + std::to_string(transB) +
               " tC=" + std::to_string(transC) +
               " A=" + dt2s(typeA) +
               " B=" + dt2s(typeB) +
               " C=" + dt2s(typeC) +
               " D=" + dt2s(typeD) + " }";
    }
};

// ------------------------------ Plan (shared) --------------------------------
enum class BackendKind : uint8_t { WMMA, Cutlass, CuTe };

struct Plan {
    virtual ~Plan() = default;
    virtual BackendKind backend() const noexcept = 0;

    // Shared hyperparameters across backends
    int tile_m{128};
    int tile_n{128};
    int tile_k{32};
    int stages{3};            // pipeline stages / smem “k-tiles”
    int split_k_slices{1};    // parallel K-splitting
    bool use_tensor_cores{true};
    dim3 block_dim{256, 1, 1};  // threads per block (x,y,z)
    dim3 grid_dim{1, 1, 1};     // blocks per grid (x,y,z)

    // Optional epilogue scaling (alpha*AB + beta*C)
    std::optional<double> alpha;  // set at dispatch time if you like
    std::optional<double> beta;
};


} // namespace dphpc