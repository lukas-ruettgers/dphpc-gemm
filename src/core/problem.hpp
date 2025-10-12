// src/core/problem.hpp
#pragma once
#include <cstdint>
#include <string>
#include <optional>
#include <stdexcept>
#include <cuda_runtime.h>

#include "device_query.hpp" 

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
    bool wmma_available{true};
    bool cutlass_available{true};
    bool cute_available{true};

    // ---------------- Hardware spec (queried at runtime) ----------------
    // The full device descriptor used by planner/dispatcher/backends.
    device_query::DeviceInfo device{};

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
        std::string s = "Problem{ M=" + std::to_string(M) +
                        " N=" + std::to_string(N) +
                        " K=" + std::to_string(K) +
                        " tA=" + std::to_string(transA) +
                        " tB=" + std::to_string(transB) +
                        " tC=" + std::to_string(transC) +
                        " A=" + dt2s(typeA) +
                        " B=" + dt2s(typeB) +
                        " C=" + dt2s(typeC) +
                        " D=" + dt2s(typeD) + " }";

        // Append a short HW hint (optional)
        s += " | Device: cc " + std::to_string(device.sm_major) + "." + std::to_string(device.sm_minor)
           + " \"" + device.name + "\"";
        return s;
    }
};

} // namespace dphpc