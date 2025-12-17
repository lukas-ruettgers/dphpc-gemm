cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GEMM_Benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 120)

# Update CUTLASS path as needed
set(CUTLASS_DIR "/home/shgoel/dphpc/cutlass/cutlass")

# Include CUTLASS + project headers
include_directories(${CUTLASS_DIR}/include)
# include_directories(${CUTLASS_DIR}/tools/util/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/home/shgoel/dphpc/cutlass/cuda-samples/Common)

# Header-only helper library
add_library(utils INTERFACE)
target_include_directories(utils INTERFACE ${CMAKE_SOURCE_DIR}/include)

# CUTLASS benchmark
add_executable(cutlass_gemm src/cutlass_gemm.cu)
target_link_libraries(cutlass_gemm PRIVATE utils)

# WMMA benchmark
add_executable(wmma_gemm src/wmma_gemm.cu)
target_link_libraries(wmma_gemm PRIVATE utils)

# WMMA optimized benchmark
add_executable(wmma_opti_gemm src/cudaTensorCoreGemm.cu)
# target_link_libraries(wmma_gemm PRIVATE utils)

# CUTLASS tunable benchmark
add_executable(cutlass_gemm_tunable src/cutlass_gemm_tunable2.cu)
target_link_libraries(cutlass_gemm_tunable PRIVATE utils)

# Optional main driver (no need to link utils)
add_executable(main src/main.cpp)
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)
