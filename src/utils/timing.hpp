// utils/timing.hpp
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <utility>
#include <algorithm>
#include "cuda_check.hpp"

namespace dphpc::timing {
// Simple RAII CUDA event wrapper
struct Event {
    cudaEvent_t ev{};
    explicit Event(unsigned flags = cudaEventDefault) {
        dphpc::cudacheck::CUDA_CHECK(cudaEventCreateWithFlags(&ev, flags));
    }
    ~Event() { cudaEventDestroy(ev); }
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

    void record_start() { dphpc::cudacheck::CUDA_CHECK(cudaEventRecord(start.ev, stream)); }
    void record_stop()  { dphpc::cudacheck::CUDA_CHECK(cudaEventRecord(stop.ev,  stream)); }
    float elapsed_ms()  {
        dphpc::cudacheck::CUDA_CHECK(cudaEventSynchronize(stop.ev));
        float ms = 0.0f;
        dphpc::cudacheck::CUDA_CHECK(cudaEventElapsedTime(&ms, start.ev, stop.ev));
        return ms;
    }
};

// Run `fn()` on `stream` with warmups & repeats; return {avg_ms, std_ms}
template <class F>
std::pair<double,double> benchmark(cudaStream_t stream, F&& fn,
                                   int warmup = 5, int iters = 50) {
    // Warmup (not timed)
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    dphpc::cudacheck::CUDA_CHECK(cudaStreamSynchronize(stream));
    dphpc::cudacheck::CUDA_CHECK_LAST();

    std::vector<double> ms;
    ms.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        CudaTimer t(stream);
        t.record_start();
        fn();                       // launch work onto the stream
        t.record_stop();
        ms.push_back(t.elapsed_ms());
    }
    // Optional: ensure all work done
    dphpc::cudacheck::CUDA_CHECK(cudaStreamSynchronize(stream));
    dphpc::cudacheck::CUDA_CHECK_LAST();

    // Robust summary (drop top/bottom 5% if enough samples)
    if (iters >= 20) {
        auto tmp = ms;
        std::sort(tmp.begin(), tmp.end());
        size_t drop = static_cast<size_t>(iters * 0.05);
        tmp.erase(tmp.begin(), tmp.begin() + drop);
        tmp.erase(tmp.end() - drop, tmp.end());
        ms.swap(tmp);
    }

    double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
    double var  = 0.0;
    for (double x : ms) var += (x - mean) * (x - mean);
    var /= (ms.size() > 1 ? (ms.size() - 1) : 1);
    return {mean, std::sqrt(var)};
}

}