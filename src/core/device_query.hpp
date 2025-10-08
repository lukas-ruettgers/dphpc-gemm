#pragma once
#include <cuda_runtime.h>
#include <string>
#include <cstdint>

namespace gemm::device_query {

enum class Arch : int { SM70 = 70, SM75 = 75, SM80 = 80, SM86 = 86, SM89 = 89, SM90 = 90, Unknown = 0 };

struct DeviceInfo {
    // Identity
    int          device_id         = 0;
    std::string  name;
    int          driver_version    = 0;   // e.g. 12040
    int          runtime_version   = 0;   // e.g. 12030

    // SM & core properties
    int          sm_major          = 0;
    int          sm_minor          = 0;
    Arch         arch              = Arch::Unknown;
    int          multiprocessors   = 0;
    int          warp_size         = 32;
    int          max_threads_per_sm= 0;
    int          max_threads_per_block = 0;

    // Registers / Shared memory limits
    int          regs_per_block    = 0;
    int          regs_per_multiprocessor = 0;
    int          regs_per_thread   = 0;   // architectural limit
    int          shared_mem_per_block = 0;          // default limit (bytes)
    int          shared_mem_per_block_optin = 0;    // opt-in limit (bytes, SM80+)
    int          shared_mem_per_sm  = 0;            // bytes per SM
    int          shared_mem_bank_size = 32;         // bytes (typical)

    // Caches
    int          l2_cache_bytes    = 0;

    // Memory & bandwidth
    size_t       global_mem_bytes  = 0;
    int          memory_clock_khz  = 0;
    int          memory_bus_width_bits = 0;
    double       theoretical_mem_bw_GBps = 0.0;

    // Features
    bool         concurrent_kernels = false;
    bool         cooperative_launch  = false;
    bool         cooperative_multi   = false;
    bool         pageable_mem_access = false;

    // Tensor core & math modes (heuristics by arch)
    bool         has_tensor_cores   = false;  // FP16/TF32/bf16/FP8 depending on arch
    bool         supports_tf32      = false;  // SM80+
    bool         supports_bf16      = false;  // SM80+
    bool         supports_fp8       = false;  // SM90+
    bool         supports_cp_async  = false;  // SM80+

    // Convenience
    int          max_blocks_per_sm  = 0;      // theoretical attribute
};

// Query the *current* CUDA device (or pass an explicit device id).
const DeviceInfo& query(int device_id = -1);

// Optional pretty-printer (useful in eval/)
std::string to_string(const DeviceInfo& d);

} // namespace gemm::device_query
