#pragma once
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

// CUTE
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm80.hpp>

namespace gemm_cpasync_pipelined {

using namespace cute;
using cutlass::half_t;

// Multi-stage pipelined with cp.async - true overlap
template<int BM, int BN, int BK, int NUM_STAGES = 3>
__launch_bounds__(256, 4)
__global__ void gemm_kernel_cpasync_pipelined(
    half_t const* __restrict__ A,
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

  constexpr int PAD_A = 8;
  extern __shared__ __align__(128) half_t smem[];
  
  constexpr int sA_size = BM * (BK + PAD_A);
  constexpr int sB_size = BN * BK;
  constexpr int stage_size = sA_size + sB_size;

  // MMA setup
  auto tiled_mma = make_tiled_mma(
      SM80_16x8x8_F32F16F16F32_TN{},
      Layout<Shape<_4,_2,_1>>{});
  
  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

  // cp.async copy atoms
  using CopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>;
  using CopyAtomB = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, half_t>;

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

  // Partition global memory
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  Tensor tCrC = thr_mma.partition_fragment_C(gC);
  Tensor tCgC = thr_mma.partition_C(gC);
  clear(tCrC);

  const int Ktiles = size<3>(tAgA);
  
 //prefill 
  const int prefill_stages = NUM_STAGES < Ktiles ? NUM_STAGES : Ktiles;

  #pragma unroll
  for (int stage = 0; stage < prefill_stages; ++stage) {
    int stage_offset = stage * stage_size;
    half_t* sA_ptr = smem + stage_offset;
    half_t* sB_ptr = sA_ptr + sA_size;
    
    auto sA = make_tensor(make_smem_ptr(sA_ptr),
                          make_shape(Int<BM>{}, Int<BK>{}),
                          make_stride(Int<BK + PAD_A>{}, Int<1>{}));
    auto sB = make_tensor(make_smem_ptr(sB_ptr),
                          make_shape(Int<BN>{}, Int<BK>{}),
                          make_stride(Int<BK>{}, Int<1>{}));
    
    Tensor tAsA = thr_copy_a.partition_D(sA);
    Tensor tBsB = thr_copy_b.partition_D(sB);
    
    copy(copy_a, tAgA(_,_,_,stage), tAsA);
    copy(copy_b, tBgB(_,_,_,stage), tBsB);
    cp_async_fence();
  }
 //main loop 
  for (int kt = 0; kt < Ktiles; ++kt) {
    int read_stage = kt % NUM_STAGES;
    int read_offset = read_stage * stage_size;
    
    half_t* sA_read_ptr = smem + read_offset;
    half_t* sB_read_ptr = sA_read_ptr + sA_size;
    
    auto sA_read = make_tensor(make_smem_ptr(sA_read_ptr),
                               make_shape(Int<BM>{}, Int<BK>{}),
                               make_stride(Int<BK + PAD_A>{}, Int<1>{}));
    auto sB_read = make_tensor(make_smem_ptr(sB_read_ptr),
                               make_shape(Int<BN>{}, Int<BK>{}),
                               make_stride(Int<BK>{}, Int<1>{}));
    
    auto tCsA = thr_mma.partition_A(sA_read);
    auto tCsB = thr_mma.partition_B(sB_read);
    
    cp_async_wait<NUM_STAGES - 1>();
    __syncthreads();
    
    // Compute on current stage while next stages are loading
    gemm(tiled_mma, tCsA, tCsB, tCrC);
    
    // Issue load for next tile (if available)
    int next_kt = kt + NUM_STAGES;
    if (next_kt < Ktiles) {
      int write_stage = next_kt % NUM_STAGES;
      int write_offset = write_stage * stage_size;
      
      half_t* sA_write_ptr = smem + write_offset;
      half_t* sB_write_ptr = sA_write_ptr + sA_size;
      
      auto sA_write = make_tensor(make_smem_ptr(sA_write_ptr),
                                  make_shape(Int<BM>{}, Int<BK>{}),
                                  make_stride(Int<BK + PAD_A>{}, Int<1>{}));
      auto sB_write = make_tensor(make_smem_ptr(sB_write_ptr),
                                  make_shape(Int<BN>{}, Int<BK>{}),
                                  make_stride(Int<BK>{}, Int<1>{}));
      
      Tensor tAsA_write = thr_copy_a.partition_D(sA_write);
      Tensor tBsB_write = thr_copy_b.partition_D(sB_write);
      
      // Issue next async load (overlaps with next iteration's compute)
      copy(copy_a, tAgA(_,_,_,next_kt), tAsA_write);
      copy(copy_b, tBgB(_,_,_,next_kt), tBsB_write);
      cp_async_fence();
    }
    
  }
cp_async_wait<0>();
__syncthreads();
  auto rC = coalesce(tCrC);
  auto gC_out = coalesce(tCgC);
  for (int i = 0; i < size(rC); ++i) {
    gC_out(i) = rC(i);
  }
}

template<int BM=128, int BN=64, int BK=64, int NUM_STAGES=3>
inline void gemm_cpasync_pipelined_launch(
    const half_t *dA,
    const half_t *dB,
    float *dC,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    cudaStream_t stream = 0)
{
  dim3 block(256);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  
  constexpr int PAD_A = 8;
  constexpr int sA_size = BM * (BK + PAD_A);
  constexpr int sB_size = BN * BK;
  constexpr int stage_size = sA_size + sB_size;
  size_t smem_bytes = NUM_STAGES * stage_size * sizeof(half_t);
//  size_t smem_bytes = NUM_STAGES * ((BM*(BK+PAD_A)) + (BN*BK)) * sizeof(half_t);
  cudaFuncSetAttribute(
      gemm_kernel_cpasync_pipelined<BM, BN, BK, NUM_STAGES>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smem_bytes);

  gemm_kernel_cpasync_pipelined<BM, BN, BK, NUM_STAGES><<<grid, block, smem_bytes, stream>>>(
      dA, dB, dC, M, N, K, lda, ldb, ldc);
}

} // namespace gemm_cpasync_pipelined
