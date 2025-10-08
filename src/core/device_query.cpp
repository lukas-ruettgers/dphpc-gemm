#include "device_query.hpp"
#include "../utils/cuda_check.hpp"   // uses dphpc::cudacheck::CUDA_CHECK(...)
#include <cuda_runtime_api.h>
#include <sstream>
#include <mutex>
#include <array>

namespace dphpc::device_query {

static Arch sm_to_arch(int major, int minor) {
    const int cc = major * 10 + minor;
    if      (cc >= 90) return Arch::SM90;
    else if (cc >= 89) return Arch::SM89;
    else if (cc >= 86) return Arch::SM86;
    else if (cc >= 80) return Arch::SM80;
    else if (cc >= 75) return Arch::SM75;
    else if (cc >= 70) return Arch::SM70;
    return Arch::Unknown;
}

static double compute_theoretical_bw_gbps(int mem_clock_khz, int bus_width_bits) {
    // DDR effective rate: 2 transfers per clock. memoryClockRate is in kHz.
    // GB/s = 2 * clock(Hz) * (bus_width_bits/8 bytes) / 1e9
    const double clock_hz = static_cast<double>(mem_clock_khz) * 1000.0;
    const double bytes_per_cycle = static_cast<double>(bus_width_bits) / 8.0;
    return 2.0 * clock_hz * bytes_per_cycle / 1e9;
}

static void fill_feature_flags(DeviceInfo& d) {
    const int cc = d.sm_major * 10 + d.sm_minor;
    d.has_tensor_cores  = (cc >= 70);  // Volta+
    d.supports_tf32     = (cc >= 80);  // Ampere+
    d.supports_bf16     = (cc >= 80);  // Ampere+
    d.supports_fp8      = (cc >= 90);  // Hopper+
    d.supports_cp_async = (cc >= 80);  // cp.async (Ampere+)
}

static void query_core(cudaDeviceProp& props, DeviceInfo& d) {
    d.name                = props.name;
    d.sm_major            = props.major;
    d.sm_minor            = props.minor;
    d.arch                = sm_to_arch(props.major, props.minor);
    d.multiprocessors     = props.multiProcessorCount;
    d.warp_size           = props.warpSize;
    d.max_threads_per_sm  = props.maxThreadsPerMultiProcessor;
    d.max_threads_per_block = props.maxThreadsPerBlock;

    d.global_mem_bytes    = props.totalGlobalMem;
    d.memory_clock_khz    = props.memoryClockRate;
    d.memory_bus_width_bits = props.memoryBusWidth;
    d.theoretical_mem_bw_GBps = compute_theoretical_bw_gbps(d.memory_clock_khz, d.memory_bus_width_bits);

    d.shared_mem_per_block       = props.sharedMemPerBlock;
    d.shared_mem_per_sm          = props.sharedMemPerMultiprocessor;
    d.regs_per_block             = props.regsPerBlock;
    d.l2_cache_bytes             = props.l2CacheSize;

    // Some values aren’t in cudaDeviceProp; query with cudaDeviceGetAttribute
    int attr = 0;

    if (cudaDeviceGetAttribute(&attr, cudaDevAttrMaxSharedMemoryPerBlockOptin, d.device_id) == cudaSuccess) {
        d.shared_mem_per_block_optin = attr;
    } else {
        d.shared_mem_per_block_optin = d.shared_mem_per_block; // fallback
    }

    if (cudaDeviceGetAttribute(&attr, cudaDevAttrMaxRegistersPerMultiprocessor, d.device_id) == cudaSuccess) {
        d.regs_per_multiprocessor = attr;
    }
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrMaxRegistersPerBlock, d.device_id) == cudaSuccess) {
        d.regs_per_block = attr; // prefer attribute when available
    }
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrMaxRegistersPerThread, d.device_id) == cudaSuccess) {
        d.regs_per_thread = attr;
    }
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrMaxBlocksPerMultiprocessor, d.device_id) == cudaSuccess) {
        d.max_blocks_per_sm = attr;
    }
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentKernels, d.device_id) == cudaSuccess) {
        d.concurrent_kernels = (attr != 0);
    }
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrCooperativeLaunch, d.device_id) == cudaSuccess) {
        d.cooperative_launch = (attr != 0);
    }
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrCooperativeMultiDeviceLaunch, d.device_id) == cudaSuccess) {
        d.cooperative_multi = (attr != 0);
    }
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrPageableMemoryAccess, d.device_id) == cudaSuccess) {
        d.pageable_mem_access = (attr != 0);
    }
    // Shared memory bank size (typical 32B; not always surfaced—keep heuristic default)
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrMaxSharedMemoryPerMultiprocessor, d.device_id) == cudaSuccess) {
        d.shared_mem_per_sm = attr; // prefer attribute path
    }
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
