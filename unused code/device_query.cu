#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Number of CUDA devices: " << deviceCount << "\n\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        int clockKHz = 0;
        cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, i);

        int memClockKHz = 0;
        cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes\n";
        std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
        std::cout << "  Warp size: " << prop.warpSize << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Number of SMs: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << "\n";
        std::cout << "  Max grid dimensions: ["
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << "]\n";
        std::cout << "  Max threads dim (block): ["
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << "]\n";
        std::cout << "  SM Clock Rate: " << clockKHz / 1000 << " MHz\n";
        std::cout << "  Memory Clock Rate: " << memClockKHz / 1000 << " MHz\n";
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
        std::cout << "  Registers per SM: " << prop.regsPerMultiprocessor << " registers\n";
        std::cout << "  Registers per Block: " << prop.regsPerBlock << "\n";
        std::cout << "  Warp size: " << prop.warpSize << "\n\n";
    }

    return 0;
}
