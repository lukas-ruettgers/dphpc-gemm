#pragma once
#include <vector>
#include <random>
#include <type_traits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <algorithm> // std::swap, std::fill_n
#include <cmath>     // std::nextafter

#include "../globals.h" // defines GLOBAL_SEED (std::uint64_t)

namespace dphpc::data {

enum class DistributionType : uint8_t { Uniform, Gaussian };

// Optional: per-stream seed mixing (e.g., A/B/C need different draws)
std::uint64_t mixed_seed(std::uint64_t base, std::uint64_t stream_id) noexcept;

// -----------------------------------------------------------------------------
// Fill an existing contiguous (row-major) HxW buffer with random values.
// -----------------------------------------------------------------------------
template <typename T>
void fillRandomMatrix(T* dst, int H, int W,
                      DistributionType distType = DistributionType::Gaussian,
                      std::uint64_t seed = GLOBAL_SEED,
                      T param1 = T(0),          // mean (Gaussian) or lower (Uniform)
                      T param2 = T(1))          // stddev (Gaussian) or upper (Uniform)
{
    static_assert(std::is_floating_point_v<T>, "T must be floating-point.");
    if (!dst) throw std::invalid_argument("fillRandomMatrix: dst is null.");
    if (H < 0 || W < 0) throw std::invalid_argument("fillRandomMatrix: H and W must be >= 0.");

    const auto h = static_cast<std::size_t>(H);
    const auto w = static_cast<std::size_t>(W);
    const auto n = h * w;
    if (w != 0 && n / w != h) throw std::overflow_error("fillRandomMatrix: H*W overflow.");
    if (n == 0) return;

    // Stable & reproducible RNG from 64-bit seed
    auto make_rng = [](std::uint64_t s) {
        std::seed_seq seq{
            static_cast<std::uint32_t>(s),
            static_cast<std::uint32_t>(s >> 32),
            0x85EBCA6Bu, 0xC2B2AE35u
        };
        return std::mt19937_64(seq);
    };
    auto rng = make_rng(seed);

    switch (distType) {
        case DistributionType::Uniform: {
            // param1 = lower, param2 = upper (swap if needed)
            T lo = param1, hi = param2;
            if (!(lo <= hi)) std::swap(lo, hi);
            // [lo, hi) by nudging upper to the next representable
            const T upper = std::nextafter(hi, std::numeric_limits<T>::max());
            std::uniform_real_distribution<T> dist(lo, upper);
            for (std::size_t i = 0; i < n; ++i) dst[i] = dist(rng);
            break;
        }
        case DistributionType::Gaussian: {
            // param1 = mean, param2 = stddev
            if (param2 == T(0)) { std::fill_n(dst, n, param1); break; }
            if (!(param2 > T(0))) throw std::invalid_argument("fillRandomMatrix: stddev must be >= 0.");
            std::normal_distribution<T> dist(param1, param2);
            for (std::size_t i = 0; i < n; ++i) dst[i] = dist(rng);
            break;
        }
        default:
            throw std::invalid_argument("fillRandomMatrix: unknown DistributionType.");
    }
}

// -----------------------------------------------------------------------------
// Allocate a vector and delegate to fillRandomMatrix (no duplication).
// -----------------------------------------------------------------------------
template <typename T>
std::vector<T> generateRandomMatrix(int H, int W,
                                    DistributionType distType = DistributionType::Gaussian,
                                    std::uint64_t seed = GLOBAL_SEED,
                                    T param1 = T(0),
                                    T param2 = T(1))
{
    static_assert(std::is_floating_point_v<T>, "T must be floating-point.");
    if (H < 0 || W < 0) throw std::invalid_argument("generateRandomMatrix: H and W must be >= 0.");

    const auto n = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
    if (W != 0 && n / static_cast<std::size_t>(W) != static_cast<std::size_t>(H))
        throw std::overflow_error("generateRandomMatrix: H*W overflow.");

    std::vector<T> y(n);
    if (n) {
        fillRandomMatrix(y.data(), H, W, distType, seed, param1, param2);
    }
    return y;
}

// Column-major index helper: elem(i,j) at i + j*ld
inline std::size_t idx_cm(int i, int j, int ld) {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

} // namespace dphpc::data
