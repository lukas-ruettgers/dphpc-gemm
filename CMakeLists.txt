cmake_minimum_required(VERSION 3.25)

project(dphpc_gemm LANGUAGES CXX CUDA)

# --- Language standards (no extra flags) -------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- External headers (header-only deps) -------------------------------------
set(CUTLASS_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/external/cutlass/include"
  "${CMAKE_SOURCE_DIR}/external/cutlass/tools/util/include"
)
# If you also vendor CuTe separately, add it here:
# set(CUTE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/cute/include")

# --- Core library -------------------------------------------------------------
add_library(gemm_core
  src/core/device_query.cpp
  src/core/dispatcher.cpp
  src/utils/data.cpp
)
target_include_directories(gemm_core
  PUBLIC  ${CMAKE_SOURCE_DIR}/src
  PRIVATE ${CUTLASS_INCLUDE_DIRS} ${CUTE_INCLUDE_DIR}
)

# --- Backend: CuTe ------------------------------------------------------------
add_library(gemm_backend_cute
  src/backend/cute/backend_adaptor.cpp
  src/backend/cute/gemm.cu
)
# Enable device linking when there are CUDA kernels
set_target_properties(gemm_backend_cute PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(gemm_backend_cute
  PUBLIC  ${CMAKE_SOURCE_DIR}/src
  PRIVATE ${CUTLASS_INCLUDE_DIRS} ${CUTE_INCLUDE_DIR}
)
target_link_libraries(gemm_backend_cute PRIVATE gemm_core)

# --- Executable (entry point: src/core/problem.cpp) --------------------------
add_executable(gemm
  src/core/problem.cpp
)
target_include_directories(gemm PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(gemm PRIVATE gemm_core gemm_backend_cute)

# Optional: make explicit (default is ${CMAKE_BINARY_DIR})
set_target_properties(gemm PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
