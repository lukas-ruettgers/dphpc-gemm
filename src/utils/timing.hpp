#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <utility>
#include <algorithm>
#include "cuda_check.hpp"

namespace dphpc::timing {

// RAII CUDA event
struct Event {
    cudaEvent_t ev{};
    explicit Event(unsigned flags = cudaEventDefault) {
        DPHPC_CUDA_CHECK(cudaEventCreateWithFlags(&ev, flags));
    }
    ~Event() { if (ev) cudaEventDestroy(ev); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& o) noexcept : ev(o.ev) { o.ev = nullptr; }
    Event& operator=(Event&& o) noexcept { std::swap(ev, o.ev); return *this; }
};

// Measures elapsed time (ms) on a stream using events
struct CudaTimer {
    cudaStream_t stream{};
    Event start, stop;

    explicit CudaTimer(cudaStream_t s = 0)
        : stream(s), start(cudaEventDefault), stop(cudaEventDefault) {}

    void record_start() { DPHPC_CUDA_CHECK(cudaEventRecord(start.ev, stream)); }
    void record_stop()  { DPHPC_CUDA_CHECK(cudaEventRecord(stop.ev,  stream)); }
    float elapsed_ms()  {
        DPHPC_CUDA_CHECK(cudaEventSynchronize(stop.ev));
        float ms = 0.0f;
        DPHPC_CUDA_CHECK(cudaEventElapsedTime(&ms, start.ev, stop.ev));
        return ms;
    }
};

// Compute mean/std (optionally trimmed)
inline std::pair<double,double> summarize(std::vector<double> samples, double trim_frac = 0.05) {
    if (samples.empty()) return {0.0, 0.0};
    if (samples.size() >= 20 && trim_frac > 0.0) {
        std::sort(samples.begin(), samples.end());
        size_t drop = static_cast<size_t>(samples.size() * trim_frac);
        if (drop * 2 < samples.size()) {
            samples.erase(samples.begin(), samples.begin() + drop);
            samples.erase(samples.end() - drop, samples.end());
        }
    }
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double var  = 0.0;
    for (double x : samples) var += (x - mean) * (x - mean);
    var /= (samples.size() > 1 ? (samples.size() - 1) : 1);
    return {mean, std::sqrt(var)};
}

} // namespace dphpc::timing
