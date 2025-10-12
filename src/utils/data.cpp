#include "data.hpp"

namespace dphpc::data {

// SplitMix64 mixer for reproducible stream separation
static inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

std::uint64_t mixed_seed(std::uint64_t base, std::uint64_t stream_id) noexcept {
    return splitmix64(base ^ (stream_id + 0xD6E8FEB86659FD93ull));
}

} // namespace dphpc::data
