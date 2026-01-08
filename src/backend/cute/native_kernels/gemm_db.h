#pragma once
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

// CUTE
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

namespace gemm_cpasync_fixed {

using namespace cute;
using cutlass::half_t;

// Double-buffered with cp.async
template<int BM, int BN, int BK>
__launch_bounds__(256, 1)
__global__ void gemm_kernel_cpasync(half_t const* __restrict__ A,
                                    half_t const* __restrict__ B,
                                    float*        __restrict__ C,
                                    int M, int N, int K,
                                    int lda, int ldb, int ldc)
{
  // Global tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(lda, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), make_stride(ldb, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(ldc, Int<1>{}));

  // Block coordinates
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Tile
  auto cta_tiler = make_shape(Int<BM>{}, Int<BN>{}, Int<BK>{});
  auto cta_coord = make_coord(by, bx, _);

  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  // Shared memory - DOUBLE BUFFER
  constexpr int PAD_A = 8;
  extern __shared__ __align__(16) half_t smem[];
  
  half_t* sA_ptr = smem;
  half_t* sB_ptr = sA_ptr + 2 * (BM * (BK + PAD_A));

  // Create TWO separate shared memory buffers (not a single tensor with extra dimension)
  auto sA0 = make_tensor(make_smem_ptr(sA_ptr),
                         make_shape(Int<BM>{}, Int<BK>{}),
                         make_stride(Int<BK + PAD_A>{}, Int<1>{}));
  auto sA1 = make_tensor(make_smem_ptr(sA_ptr + BM * (BK + PAD_A)),
                         make_shape(Int<BM>{}, Int<BK>{}),
                         make_stride(Int<BK + PAD_A>{}, Int<1>{}));
  
  auto sB0 = make_tensor(make_smem_ptr(sB_ptr),
                         make_shape(Int<BN>{}, Int<BK>{}),
                         make_stride(Int<BK>{}, Int<1>{}));
  auto sB1 = make_tensor(make_smem_ptr(sB_ptr + BN * BK),
                         make_shape(Int<BN>{}, Int<BK>{}),
                         make_stride(Int<BK>{}, Int<1>{}));

  // MMA setup
  auto tiled_mma = make_tiled_mma(
      SM80_16x8x16_F32F16F16F32_TN{},
      Layout<Shape<_4,_2,_1>>{});
  
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

  // cp.async copy
  using TA = half_t;
  using TB = half_t;
  using CopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, TA>;
  using CopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, TB>;

  auto copy_a = make_tiled_copy(
      CopyAtomA{}, 
      Layout<Shape<_32,_8>, Stride<_8,_1>>{},
      Layout<Shape<_1,_8>>{});
  
  auto copy_b = make_tiled_copy(
      CopyAtomB{}, 
      Layout<Shape<_32,_8>, Stride<_8,_1>>{},
      Layout<Shape<_1,_8>>{});

  auto thr_copy_a = copy_a.get_thread_slice(threadIdx.x);
  auto thr_copy_b = copy_b.get_thread_slice(threadIdx.x);

  // Partition for both buffers
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  
  Tensor tAsA0 = thr_copy_a.partition_D(sA0);
  Tensor tAsA1 = thr_copy_a.partition_D(sA1);
  Tensor tBsB0 = thr_copy_b.partition_D(sB0);
  Tensor tBsB1 = thr_copy_b.partition_D(sB1);

  Tensor tCrC = thr_mma.partition_fragment_C(gC);
  Tensor tCgC = thr_mma.partition_C(gC);
  clear(tCrC);

  auto tCsA0 = thr_mma.partition_A(sA0);
  auto tCsA1 = thr_mma.partition_A(sA1);
  auto tCsB0 = thr_mma.partition_B(sB0);
  auto tCsB1 = thr_mma.partition_B(sB1);

  const int Ktiles = size<3>(tAgA);
  
  // Prologue: Load first tile into buffer 0
  if (Ktiles > 0) {
    copy(copy_a, tAgA(_,_,_,0), tAsA0);
    copy(copy_b, tBgB(_,_,_,0), tBsB0);
    cp_async_fence();
  }

  // Main K loop - DOUBLE BUFFERED
  for (int kt = 0; kt < Ktiles; ++kt)
  {
    int read_stage = kt % 2;

    // Prefetch next tile (if exists) into the OTHER buffer
    if (kt + 1 < Ktiles) {
      if (read_stage == 0) {
        copy(copy_a, tAgA(_,_,_,kt+1), tAsA1);
        copy(copy_b, tBgB(_,_,_,kt+1), tBsB1);
      } else {
        copy(copy_a, tAgA(_,_,_,kt+1), tAsA0);
        copy(copy_b, tBgB(_,_,_,kt+1), tBsB0);
      }
      cp_async_fence();
    }

    // Wait for current tile to complete
    cp_async_wait<0>();
    __syncthreads();

    // Compute on read_stage
    if (read_stage == 0) {
      gemm(tiled_mma, tCsA0, tCsB0, tCrC);
    } else {
      gemm(tiled_mma, tCsA1, tCsB1, tCrC);
    }
    __syncthreads();
  }

  // Store
  auto rC = coalesce(tCrC);
  auto gC_out = coalesce(tCgC);
  for (int i = 0; i < size(rC); ++i) {
    gC_out(i) = rC(i);
  }
}

template<int BM=128, int BN=64, int BK=64>
inline void gemm_cpasync_launch(const half_t *dA,
                                const half_t *dB,
                                float *dC,
                                int M, int N, int K,
                                int lda, int ldb, int ldc,
                                cudaStream_t stream = 0)
{
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  constexpr int PAD_A = 8;
  size_t smem_bytes = 2 * (BM * (BK + PAD_A) + BN * BK) * sizeof(half_t);

  cudaFuncSetAttribute(
      gemm_kernel_cpasync<BM,BN,BK>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes);

  gemm_kernel_cpasync<BM,BN,BK><<<grid, block, smem_bytes, stream>>>(
      dA, dB, dC, M, N, K, lda, ldb, ldc);
}

} // namespace gemm_cpasync_fixed
