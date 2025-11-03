#pragma once
#include <iostream>
#include <functional>
#include <cuda_runtime.h>

#define CUDA_CHECK(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

inline double time_cuda_event(std::function<void()> func, int warmup = 2, int repeat = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup; ++i) func();

    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repeat;
}
