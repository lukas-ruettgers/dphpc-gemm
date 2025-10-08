// utils/cuda_check.hpp
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace dphpc::cudacheck {
// Convert cudaError_t to exception with file:line context
inline void __cudaCheck(cudaError_t err, const char* expr,
                        const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line
            << " (" << expr << "): "
            << cudaGetErrorString(err) << " [" << static_cast<int>(err) << "]";
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) __cudaCheck((expr), #expr, __FILE__, __LINE__)

// Check for async kernel errors without a sync barrier
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Optional: sync and check (use sparingly in perf paths)
#define CUDA_SYNC_CHECK(stream) \
    do { \
        CUDA_CHECK(cudaStreamSynchronize((stream))); \
        CUDA_CHECK_LAST(); \
    } while (0)

// Scoped device setter + reset (RAII)
struct CudaDeviceGuard {
    int prev{-1};
    explicit CudaDeviceGuard(int dev) {
        CUDA_CHECK(cudaGetDevice(&prev));
        if (dev != prev) CUDA_CHECK(cudaSetDevice(dev));
    }
    ~CudaDeviceGuard() noexcept(false) {
        if (prev >= 0) CUDA_CHECK(cudaSetDevice(prev));
    }
};

// Align value up to multiple (useful for launch params)
template <typename T>
constexpr T align_up(T x, T a) { return (x + (a - 1)) / a * a; }
}
