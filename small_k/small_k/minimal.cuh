#pragma once

#include <cuda_runtime.h>
#include <algorithm>

// CUTLASS/CUTE
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

namespace gemm_cute_min {

using namespace cute;
using cutlass::half_t;


#ifndef CHECK_CUDA
#define CHECK_CUDA(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)
#endif



// -------------------- Tensor Core FP16xFP16->FP32 (SM80+, TN) ----------------


#ifndef BM_TC
#define BM_TC 64
#endif

#ifndef BN_TC
#define BN_TC 64
#endif

#ifndef BK_TC
#define BK_TC 32
#endif

// -------------------- Tensor Core FP16xFP16 -> FP32 (TN layout) -----------------------

__launch_bounds__(256, 1,1)
__global__ void gemm_kernel_tc_fp16acc32_TN_cpasync(
    half_t const* __restrict__ A,
    half_t const* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  
  const int m0 = by * BM_TC;
  const int n0 = bx * BN_TC;
  
  if (m0 >= M || n0 >= N) return;

  // Global tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride( Int<1>{}, lda));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(ldb, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(ldc, Int<1>{}));

  // Shared memory
  constexpr int PAD_A = 8;
  extern __shared__ __align__(16) unsigned char smem_tc_tn[];
  half_t* sA_ptr = reinterpret_cast<half_t*>(smem_tc_tn);
  half_t* sB_ptr = reinterpret_cast<half_t*>(smem_tc_tn + sizeof(half_t) * (size_t)BM_TC * (BK_TC + PAD_A));

  auto sA = make_tensor(make_smem_ptr(sA_ptr),
                        make_shape(Int<BM_TC>{}, Int<BK_TC>{}),
                        make_stride(Int<BK_TC + PAD_A>{}, Int<1>{}));

  auto sB = make_tensor(make_smem_ptr(sB_ptr),
                        make_shape(Int<BN_TC>{}, Int<BK_TC>{}),
                        make_stride(Int<BK_TC>{}, Int<1>{}));

  // MMA setup
  auto tiled_mma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<_2,_4,_1>>{});
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

  // C fragments
  Tensor gC_block = make_tensor(mC.data() + m0 * ldc + n0,
                                make_shape(Int<BM_TC>{}, Int<BN_TC>{}),
                                make_stride(ldc, Int<1>{}));
  
  Tensor tCgC = thr_mma.partition_C(gC_block);
  Tensor tCrC = thr_mma.partition_fragment_C(gC_block);
  clear(tCrC);

  // Partition shared for MMA
  auto tAsA_mma = thr_mma.partition_A(sA);
  auto tBsB_mma = thr_mma.partition_B(sB);

  // Main K loop with VECTORIZED loads
  const int num_k_tiles = (K + BK_TC - 1) / BK_TC;
  
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    const int k0 = kt * BK_TC;
    
    // Zero shared memory
    for (int i = threadIdx.x; i < BM_TC * (BK_TC + PAD_A); i += blockDim.x) {
      sA_ptr[i] = half_t(0);
    }
    for (int i = threadIdx.x; i < BN_TC * BK_TC; i += blockDim.x) {
      sB_ptr[i] = half_t(0);
    }
    __syncthreads();

    // VECTORIZED Load A: Load 8 half_t (16 bytes) at a time using float4
    constexpr int VEC_SIZE = 8;  // 8 half_t = 16 bytes
    const int num_vec_loads_A = (BM_TC * BK_TC) / VEC_SIZE;
    
    for (int i = threadIdx.x; i < num_vec_loads_A; i += blockDim.x) {
      int elem_idx = i * VEC_SIZE;
      int local_m = elem_idx / BK_TC;
      int local_k = elem_idx % BK_TC;
      int global_m = m0 + local_m;
      int global_k = k0 + local_k;
      
      // Check if entire vector is in bounds
      if (global_m < M && global_k + VEC_SIZE - 1 < K) {
        // Vectorized load: 16 bytes (float4)
        float4 vec = *reinterpret_cast<const float4*>(
            &A[global_m * lda + global_k]);
        *reinterpret_cast<float4*>(
            &sA_ptr[local_m * (BK_TC + PAD_A) + local_k]) = vec;
      }
      else if (global_m < M && global_k < K) {
        // Scalar fallback for partial vectors at boundaries
        for (int v = 0; v < VEC_SIZE; ++v) {
          if (global_k + v < K) {
            sA_ptr[local_m * (BK_TC + PAD_A) + local_k + v] = 
                A[global_m * lda + global_k + v];
          }
        }
      }
    }

    // VECTORIZED Load B: Load 8 half_t (16 bytes) at a time using float4
    const int num_vec_loads_B = (BN_TC * BK_TC) / VEC_SIZE;
    
    for (int i = threadIdx.x; i < num_vec_loads_B; i += blockDim.x) {
      int elem_idx = i * VEC_SIZE;
      int local_n = elem_idx / BK_TC;
      int local_k = elem_idx % BK_TC;
      int global_n = n0 + local_n;
      int global_k = k0 + local_k;
      
      // Check if entire vector is in bounds
      if (global_n < N && global_k + VEC_SIZE - 1 < K) {
        // Vectorized load: 16 bytes (float4)
        float4 vec = *reinterpret_cast<const float4*>(
            &B[global_n * ldb + global_k]);
        *reinterpret_cast<float4*>(
            &sB_ptr[local_n * BK_TC + local_k]) = vec;
      }
      else if (global_n < N && global_k < K) {
        // Scalar fallback for partial vectors at boundaries
        for (int v = 0; v < VEC_SIZE; ++v) {
          if (global_k + v < K) {
            sB_ptr[local_n * BK_TC + local_k + v] = 
                B[global_n * ldb + global_k + v];
          }
        }
      }
    }
    
    __syncthreads();


    // MMA
    gemm(tiled_mma, tAsA_mma, tBsB_mma, tCrC);
    __syncthreads();
  }
    __threadfence_block();
  // Store

for (int i = 0; i < size(tCrC); ++i) {
  tCgC(i) = tCrC(i);
}

}

// -------------------- Host wrapper: fp16 GEMM (non-padded version) ----------------------

inline void gemm_cute_tc_fp16_launch(
    const half_t* dA_f16,
    const half_t* dB_f16,
    float* dC_f32,
    int M, int N, int K,
    cudaStream_t stream = 0)
{
  dim3 block(256, 1, 1);
  dim3 grid((N + BN_TC - 1) / BN_TC,
            (M + BM_TC - 1) / BM_TC, 1);

  // DOUBLE shared memory for double buffering
  constexpr int PAD_A = 8;
  size_t smem_bytes = sizeof(half_t)  * (  // 2x for double buffering!
      size_t(BM_TC) * (BK_TC + PAD_A) +
      size_t(BN_TC) * BK_TC);

  cudaFuncSetAttribute(
      gemm_kernel_tc_fp16acc32_TN_cpasync,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      int(smem_bytes));

  gemm_kernel_tc_fp16acc32_TN_cpasync<<<grid, block, smem_bytes, stream>>>(
      dA_f16, dB_f16, dC_f32,
      M, N, K, K, K, N);
}

} // namespace gemm_cute_min


