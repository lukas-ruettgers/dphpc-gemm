#include "device_query.hpp"
#include "../utils/cuda_check.hpp"
#include <cuda_runtime.h>          // broader than cuda_runtime_api.h; helps with enums
#include <sstream>
#include <mutex>
#include <array>

#ifndef CUDART_VERSION
#define CUDART_VERSION 0
#endif

namespace dphpc::device_query {

static Arch sm_to_arch(int major, int minor) {
    const int cc = major * 10 + minor;

    // Keep your existing buckets; accept future CCs gracefully.
    if      (cc >= 100) return Arch::SM90; // treat SM100+ as ">= Hopper" bucket if your enum lacks SM100
    else if (cc >= 90)  return Arch::SM90;
    else if (cc >= 89)  return Arch::SM89;
    else if (cc >= 86)  return Arch::SM86;
    else if (cc >= 80)  return Arch::SM80;
    else if (cc >= 75)  return Arch::SM75;
    else if (cc >= 70)  return Arch::SM70;
    else if (cc >= 62)  return Arch::SM70; // Pascal GP10x (61/62) – closest bucket in your enum
    else if (cc >= 60)  return Arch::SM70; // Pascal GP100 (60)
    return Arch::Unknown;
}

static double compute_theoretical_bw_gbps(int mem_clock_khz, int bus_width_bits) {
    // Effective DDR: 2 transfers per clock.
    const double clock_hz = static_cast<double>(mem_clock_khz) * 1000.0;
    const double bytes_per_cycle = static_cast<double>(bus_width_bits) / 8.0;
    return 2.0 * clock_hz * bytes_per_cycle / 1e9;
}

// Small helper: query an attribute if the header exposes it (via version checks), else default.
static int get_attr_or(int device_id, int attr_enum_value, bool try_query, int fallback) {
    if (!try_query) return fallback;
    int v = 0;
    if (cudaDeviceGetAttribute(&v, static_cast<cudaDeviceAttr>(attr_enum_value), device_id) == cudaSuccess)
        return v;
    return fallback;
}

static void fill_feature_flags(DeviceInfo& d) {
    const int cc = d.sm_major * 10 + d.sm_minor;
    d.has_tensor_cores  = (cc >= 70);  // Volta+
    d.supports_tf32     = (cc >= 80);  // Ampere+
    d.supports_bf16     = (cc >= 80);  // Ampere+
    d.supports_fp8      = (cc >= 90);  // Hopper+
    d.supports_cp_async = (cc >= 80);  // Ampere+
}

static void query_core(cudaDeviceProp& props, DeviceInfo& d) {
    // Stable basics
    d.name                   = props.name;
    d.sm_major               = props.major;
    d.sm_minor               = props.minor;
    d.arch                   = sm_to_arch(props.major, props.minor);
    d.multiprocessors        = props.multiProcessorCount;
    d.max_threads_per_sm     = props.maxThreadsPerMultiProcessor;
    d.max_threads_per_block  = props.maxThreadsPerBlock;
    d.global_mem_bytes       = props.totalGlobalMem;

    // Prefer attributes where available; fall back to props.* only for very old toolkits.
    // Warp size (attribute has been around forever)
    d.warp_size = get_attr_or(d.device_id,
                              cudaDevAttrWarpSize,
                              /*try_query=*/true,
                              /*fallback=*/props.warpSize);

    // L2 cache size
#if CUDART_VERSION >= 8000
    d.l2_cache_bytes = get_attr_or(d.device_id,
                                   cudaDevAttrL2CacheSize,
                                   /*try_query=*/true,
                                   /*fallback=*/props.l2CacheSize);
#else
    d.l2_cache_bytes = props.l2CacheSize;
#endif

    // Memory bus width (bits) and memory clock (kHz)
#if CUDART_VERSION >= 8000
    d.memory_bus_width_bits = get_attr_or(d.device_id,
                                          cudaDevAttrGlobalMemoryBusWidth,
                                          /*try_query=*/true,
                                          /*fallback=*/props.memoryBusWidth);
    d.memory_clock_khz      = get_attr_or(d.device_id,
                                          cudaDevAttrMemoryClockRate,
                                          /*try_query=*/true,
                                          /*fallback=*/
#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDACC_RTC__)
                                          props.memoryClockRate
#else
                                          0
#endif
                                          );
#else
    d.memory_bus_width_bits = props.memoryBusWidth;
# if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDACC_RTC__)
    d.memory_clock_khz      = props.memoryClockRate;
# else
    d.memory_clock_khz      = 0;
# endif
#endif

    d.theoretical_mem_bw_GBps = compute_theoretical_bw_gbps(d.memory_clock_khz, d.memory_bus_width_bits);

    // Shared memory sizes
    d.shared_mem_per_block = props.sharedMemPerBlock; // legacy name; attribute below may prefer newer values
#if CUDART_VERSION >= 8000
    d.shared_mem_per_sm    = get_attr_or(d.device_id,
                                         cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                         /*try_query=*/true,
                                         /*fallback=*/props.sharedMemPerMultiprocessor);
#else
    d.shared_mem_per_sm    = props.sharedMemPerMultiprocessor;
#endif

    // Opt-in per-block shared memory (if unavailable, fall back to regular per-block)
#if CUDART_VERSION >= 8000
    d.shared_mem_per_block_optin = get_attr_or(d.device_id,
                                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                               /*try_query=*/true,
                                               d.shared_mem_per_block);
#else
    d.shared_mem_per_block_optin = d.shared_mem_per_block;
#endif

    // Registers
#if CUDART_VERSION >= 8000
    d.regs_per_multiprocessor = get_attr_or(d.device_id,
                                            cudaDevAttrMaxRegistersPerMultiprocessor,
                                            /*try_query=*/true,
                                            /*fallback=*/0);
    // Prefer attribute for regs per block if present
    d.regs_per_block = get_attr_or(d.device_id,
                                   cudaDevAttrMaxRegistersPerBlock,
                                   /*try_query=*/true,
                                   /*fallback=*/props.regsPerBlock);
    // Per-thread register limit (guard—older toolkits may not expose)
#else
    d.regs_per_multiprocessor = 0;
    d.regs_per_block          = props.regsPerBlock;
    // leave d.regs_per_thread as default
#endif

    // Occupancy-related caps
#if CUDART_VERSION >= 8000
    d.max_blocks_per_sm = get_attr_or(d.device_id,
                                      cudaDevAttrMaxBlocksPerMultiprocessor,
                                      /*try_query=*/true,
                                      /*fallback=*/d.max_blocks_per_sm);
#endif

    // Capabilities
#if CUDART_VERSION >= 8000
    d.concurrent_kernels = (get_attr_or(d.device_id, cudaDevAttrConcurrentKernels, true, 0) != 0);
    d.cooperative_launch = (get_attr_or(d.device_id, cudaDevAttrCooperativeLaunch, true, 0) != 0);
# if CUDART_VERSION >= 9000
    d.cooperative_multi  = (get_attr_or(d.device_id, cudaDevAttrCooperativeMultiDeviceLaunch, true, 0) != 0);
# else
    d.cooperative_multi  = false;
# endif
    d.pageable_mem_access = (get_attr_or(d.device_id, cudaDevAttrPageableMemoryAccess, true, 0) != 0);
#else
    d.concurrent_kernels  = (props.asyncEngineCount > 0); // weak fallback
    d.cooperative_launch  = false;
    d.cooperative_multi   = false;
    d.pageable_mem_access = false;
#endif
}

static void query_versions(DeviceInfo& d) {
    dphpc::cudacheck::CUDA_CHECK(cudaDriverGetVersion(&d.driver_version));
    dphpc::cudacheck::CUDA_CHECK(cudaRuntimeGetVersion(&d.runtime_version));
}

static DeviceInfo do_query(int device_id) {
    DeviceInfo d{};
    int dev = device_id;
    if (dev < 0) dphpc::cudacheck::CUDA_CHECK(cudaGetDevice(&dev));
    d.device_id = dev;

    cudaDeviceProp props{};
    dphpc::cudacheck::CUDA_CHECK(cudaGetDeviceProperties(&props, dev));

    query_versions(d);
    query_core(props, d);
    fill_feature_flags(d);
    return d;
}

// Cache per device id
const DeviceInfo& query(int device_id) {
    static std::mutex mtx;
    static std::array<DeviceInfo, 64> cache{}; // up to 64 GPUs in a node
    static std::array<bool, 64>       filled{};
    int dev = device_id;
    if (dev < 0) dphpc::cudacheck::CUDA_CHECK(cudaGetDevice(&dev));

    std::scoped_lock lk(mtx);
    if (!filled[dev]) {
        cache[dev] = do_query(dev);
        filled[dev] = true;
    }
    return cache[dev];
}

std::string to_string(const DeviceInfo& d) {
    std::ostringstream oss;
    oss << "Device " << d.device_id << " \"" << d.name << "\"\n"
        << "  Compute Capability: " << d.sm_major << "." << d.sm_minor
        << " (arch " << static_cast<int>(d.arch) << ")\n"
        << "  SMs: " << d.multiprocessors
        << ", Warp size: " << d.warp_size
        << ", Max threads/block: " << d.max_threads_per_block
        << ", Max threads/SM: " << d.max_threads_per_sm << "\n"
        << "  Shared mem/block: " << d.shared_mem_per_block
        << " B (opt-in: " << d.shared_mem_per_block_optin
        << " B), Shared mem/SM: " << d.shared_mem_per_sm << " B\n"
        << "  Regs/block: " << d.regs_per_block
        << ", Regs/SM: " << d.regs_per_multiprocessor
        << ", Regs/thread limit: " << d.regs_per_thread << "\n"
        << "  L2 size: " << d.l2_cache_bytes << " B\n"
        << "  Global mem: " << d.global_mem_bytes << " B\n"
        << "  Mem clock: " << d.memory_clock_khz << " kHz"
        << ", Bus width: " << d.memory_bus_width_bits << " bits"
        << ", Theoretical BW: " << d.theoretical_mem_bw_GBps << " GB/s\n"
        << "  Concurrent kernels: " << (d.concurrent_kernels ? "yes" : "no")
        << ", Coop launch: " << (d.cooperative_launch ? "yes" : "no")
        << ", Coop multi-device: " << (d.cooperative_multi ? "yes" : "no") << "\n"
        << "  Tensor cores: " << (d.has_tensor_cores ? "yes" : "no")
        << " | TF32: " << (d.supports_tf32 ? "yes" : "no")
        << " | BF16: " << (d.supports_bf16 ? "yes" : "no")
        << " | FP8: "  << (d.supports_fp8 ? "yes" : "no")
        << " | cp.async: " << (d.supports_cp_async ? "yes" : "no") << "\n"
        << "  Driver/Runtime: " << d.driver_version << "/" << d.runtime_version;
    return oss.str();
}

} // namespace dphpc::device_query
