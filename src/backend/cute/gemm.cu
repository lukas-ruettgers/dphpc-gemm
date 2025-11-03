/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <stdexcept>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/stride.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include "plan.hpp"                    // dphpc::backend::cute::PlanCuTe + tags
#include "backend_adaptor.hpp"

namespace dphpc::backend::cute {

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__
__launch_bounds__(decltype(::cute::size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* __restrict__ A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* __restrict__ B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * __restrict__ C, CStride dC, CSmemLayout sC_layout, CThreadLayout tC,
            Alpha alpha, Beta beta)
{
  // NOTE: Added __restrict__ to A/B/C pointers to help the compiler

  // Preconditions
  CUTE_STATIC_ASSERT_V(::cute::rank(shape_MNK) == ::cute::Int<3>{});         // (M, N, K)
  CUTE_STATIC_ASSERT_V(::cute::rank(cta_tiler) == ::cute::Int<3>{});         // (BLK_M, BLK_N, BLK_K)

  static_assert(::cute::is_static<AThreadLayout>::value);
  static_assert(::cute::is_static<BThreadLayout>::value);
  static_assert(::cute::is_static<CThreadLayout>::value);

  CUTE_STATIC_ASSERT_V(::cute::size(tA) == ::cute::size(tB));                // NumThreads
  CUTE_STATIC_ASSERT_V(::cute::size(tC) == ::cute::size(tA));                // NumThreads

  CUTE_STATIC_ASSERT_V(::cute::size<0>(cta_tiler) % ::cute::size<0>(tA) == ::cute::Int<0>{});  // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(::cute::size<2>(cta_tiler) % ::cute::size<1>(tA) == ::cute::Int<0>{});  // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(::cute::size<1>(cta_tiler) % ::cute::size<0>(tB) == ::cute::Int<0>{});  // BLK_N / THR_N
  CUTE_STATIC_ASSERT_V(::cute::size<2>(cta_tiler) % ::cute::size<1>(tB) == ::cute::Int<0>{});  // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(::cute::size<0>(cta_tiler) % ::cute::size<0>(tC) == ::cute::Int<0>{});  // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(::cute::size<1>(cta_tiler) % ::cute::size<1>(tC) == ::cute::Int<0>{});  // BLK_N / THR_N

  static_assert(::cute::is_static<ASmemLayout>::value);
  static_assert(::cute::is_static<BSmemLayout>::value);
  static_assert(::cute::is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(::cute::size<0>(ASmemLayout{}) == ::cute::size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(::cute::size<0>(CSmemLayout{}) == ::cute::size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(::cute::size<0>(BSmemLayout{}) == ::cute::size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(::cute::size<1>(CSmemLayout{}) == ::cute::size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(::cute::size<1>(ASmemLayout{}) == ::cute::size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(::cute::size<1>(BSmemLayout{}) == ::cute::size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(::cute::congruent(::cute::select<0,2>(shape_MNK), dA));   // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(::cute::congruent(::cute::select<1,2>(shape_MNK), dB));   // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(::cute::congruent(::cute::select<0,1>(shape_MNK), dC));   // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  auto mA = ::cute::make_tensor(::cute::make_gmem_ptr(A), ::cute::select<0,2>(shape_MNK), dA); // (M,K)
  auto mB = ::cute::make_tensor(::cute::make_gmem_ptr(B), ::cute::select<1,2>(shape_MNK), dB); // (N,K)
  auto mC = ::cute::make_tensor(::cute::make_gmem_ptr(C), ::cute::select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = ::cute::make_coord(blockIdx.x, blockIdx.y, ::cute::_);              // (m,n,k)
  auto gA = ::cute::local_tile(mA, cta_tiler, cta_coord, ::cute::Step<::cute::_1, ::cute::X,::cute::_1>{});  // (BLK_M,BLK_K,k)
  auto gB = ::cute::local_tile(mB, cta_tiler, cta_coord, ::cute::Step< ::cute::X,::cute::_1,::cute::_1>{});  // (BLK_N,BLK_K,k)
  auto gC = ::cute::local_tile(mC, cta_tiler, cta_coord, ::cute::Step<::cute::_1,::cute::_1, ::cute::X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  __shared__ TA smemA[::cute::cosize_v<ASmemLayout>];
  __shared__ TB smemB[::cute::cosize_v<BSmemLayout>];
  auto sA = ::cute::make_tensor(::cute::make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  auto sB = ::cute::make_tensor(::cute::make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  auto tAgA = ::cute::local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
  auto tAsA = ::cute::local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

  auto tBgB = ::cute::local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
  auto tBsB = ::cute::local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

  CUTE_STATIC_ASSERT_V(::cute::size<0>(tAgA) == ::cute::size<0>(tAsA));      // THR_M
  CUTE_STATIC_ASSERT_V(::cute::size<1>(tAgA) == ::cute::size<1>(tAsA));      // THR_K
  CUTE_STATIC_ASSERT_V(::cute::size<0>(tBgB) == ::cute::size<0>(tBsB));      // THR_N
  CUTE_STATIC_ASSERT_V(::cute::size<1>(tBgB) == ::cute::size<1>(tBsB));      // THR_K

  //
  // Define A/B partitioning and C accumulators
  //

  // Partition sA (BLK_M, BLK_K) by the rows of tC
  auto tCsA = ::cute::local_partition(sA, tC, threadIdx.x, ::cute::Step<::cute::_1, ::cute::X>{});   // (THR_M,BLK_K)
  // Partition sB (BLK_N, BLK_K) by the cols of tC
  auto tCsB = ::cute::local_partition(sB, tC, threadIdx.x, ::cute::Step< ::cute::X,::cute::_1>{});   // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  auto tCgC = ::cute::local_partition(gC, tC, threadIdx.x, ::cute::Step<::cute::_1,::cute::_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  auto tCrC = ::cute::make_tensor_like(tCgC);                                // (THR_M,THR_N)

  CUTE_STATIC_ASSERT_V(::cute::size<0>(tCrC) == ::cute::size<0>(tCgC));      // THR_M
  CUTE_STATIC_ASSERT_V(::cute::size<0>(tCrC) == ::cute::size<0>(tCsA));      // THR_M
  CUTE_STATIC_ASSERT_V(::cute::size<1>(tCrC) == ::cute::size<1>(tCgC));      // THR_N
  CUTE_STATIC_ASSERT_V(::cute::size<1>(tCrC) == ::cute::size<0>(tCsB));      // THR_N
  CUTE_STATIC_ASSERT_V(::cute::size<1>(tCsA) == ::cute::size<1>(tCsB));      // BLK_K

  // Clear the accumulators
  clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

  // TUTORIAL: Example of a simple mainloop that read tiles of data into shared memory,
  //           and then computes on those tiles.
  //   copy(.) operates on the global and shared memory via the tA|tB partitioning
  //   gemm(.) operates on the shared and register memory via the tC partitioning

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy gmem to smem with tA|tB thread-partitioned tensors
    ::cute::copy(tAgA(::cute::_,::cute::_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    ::cute::copy(tBgB(::cute::_,::cute::_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

    #if __CUDA_ARCH__ >= 800
      ::cute::cp_async_fence();
      ::cute::cp_async_wait<0>();
    #else
      // noop; the preceding copies are synchronous already
    #endif

    __syncthreads();         // Wait for all threads to write to smem

    // Compute gemm on tC thread-partitioned smem
    ::cute::gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

    __syncthreads();         // Wait for all threads to read from smem
  }
#endif

  //
  // Epilogue
  //
  ::cute::axpby(alpha, tCrC, beta, tCgC);
}


//------------------------------------------------------------------------------
// Helpers: map PlanCuTe tags -> CuTe layouts
//------------------------------------------------------------------------------
template<bool Right> static inline auto make_smem_layout_A(::cute::Int bM, ::cute::Int bK) {
  if constexpr (Right)
    return ::cute::make_layout(::cute::make_shape(bM, bK), ::cute::LayoutRight{});
  else
    return ::cute::make_layout(::cute::make_shape(bM, bK));
}
template<bool Right> static inline auto make_smem_layout_B(::cute::Int bN, ::cute::Int bK) {
  if constexpr (Right)
    return ::cute::make_layout(::cute::make_shape(bN, bK), ::cute::LayoutRight{});
  else
    return ::cute::make_layout(::cute::make_shape(bN, bK));
}
template<bool Right> static inline auto make_smem_layout_C(::cute::Int bM, ::cute::Int bN) {
  if constexpr (Right)
    return ::cute::make_layout(::cute::make_shape(bM, bN), ::cute::LayoutRight{});
  else
    return ::cute::make_layout(::cute::make_shape(bM, bN));
}

template<bool Right> static inline auto make_thread_layout_A() {
  if constexpr (Right)
    return ::cute::make_layout(::cute::make_shape(::cute::Int<32>{}, ::cute::Int<8>{}), ::cute::LayoutRight{});
  else
    return ::cute::make_layout(::cute::make_shape(::cute::Int<32>{}, ::cute::Int<8>{}));
}
template<bool Right> static inline auto make_thread_layout_B() {
  if constexpr (Right)
    return ::cute::make_layout(::cute::make_shape(::cute::Int<32>{}, ::cute::Int<8>{}), ::cute::LayoutRight{});
  else
    return ::cute::make_layout(::cute::make_shape(::cute::Int<32>{}, ::cute::Int<8>{}));
}
template<bool Right> static inline auto make_thread_layout_C() {
  if constexpr (Right)
    return ::cute::make_layout(::cute::make_shape(::cute::Int<16>{}, ::cute::Int<16>{}), ::cute::LayoutRight{});
  else
    return ::cute::make_layout(::cute::make_shape(::cute::Int<16>{}, ::cute::Int<16>{}));
}

//------------------------------------------------------------------------------
// Templated tile-specialized launcher
//------------------------------------------------------------------------------
template<int BM,int BN,int BK, bool RightA,bool RightB,bool RightC, bool NTPath>
static cudaError_t
launch_for_tiles(const PlanCuTe& p,
                 char ta, char tb,
                 int m, int n, int k,
                 float alpha,
                 const float* A, int ldA,
                 const float* B, int ldB,
                 float  beta,
                 float* C, int ldC,
                 cudaStream_t stream)
{
    // Dynamic problem shape
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = ::cute::make_shape(M, N, K);             // (M,N,K)

    // CTA tile (compile-time)
    auto bM = ::cute::Int<BM>{};
    auto bN = ::cute::Int<BN>{};
    auto bK = ::cute::Int<BK>{};
    auto cta_tiler = ::cute::make_shape(bM, bN, bK);

    // Shared memory layouts
    auto sA = make_smem_layout_A<RightA>(bM, bK);
    auto sB = make_smem_layout_B<RightB>(bN, bK);
    auto sC = make_smem_layout_C<RightC>(bM, bN);
    
    // Thread layouts
    auto tA = make_thread_layout_A<RightA>();
    auto tB = make_thread_layout_B<RightB>();
    auto tC = make_thread_layout_C<RightC>();

    // Launch configuration
    dim3 dimBlock(::cute::size(tC));
    dim3 dimGrid(::cute::size(::cute::ceil_div(M, bM)),
                 ::cute::size(::cute::ceil_div(N, bN)));

    // Optional overrides from plan
    if (p.override_block_threads > 0) dimBlock.x = static_cast<unsigned>(p.override_block_threads);
    if (p.override_grid_m        > 0) dimGrid.x  = static_cast<unsigned>(p.override_grid_m);
    if (p.override_grid_n        > 0) dimGrid.y  = static_cast<unsigned>(p.override_grid_n);
    
    if constexpr (NTPath) {
      auto dA = ::cute::make_stride(::cute::Int<1>{}, ldA);
      auto dB = ::cute::make_stride(::cute::Int<1>{}, ldB);
      auto dC = ::cute::make_stride(::cute::Int<1>{}, ldC);
      gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
          prob_shape, cta_tiler,
          A, dA, sA, tA,
          B, dB, sB, tB,
          C, dC, sC, tC,
          alpha, beta);      
    }
    else{
      auto dA = ::cute::make_stride(ldA, ::cute::Int<1>{});
      auto dB = ::cute::make_stride(ldB, ::cute::Int<1>{});
      auto dC = ::cute::make_stride(::cute::Int<1>{}, ldC);
      gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
          prob_shape, cta_tiler,
          A, dA, sA, tA,
          B, dB, sB, tB,
          C, dC, sC, tC,
          alpha, beta);      
    }
    // Launch device kernel

    return cudaGetLastError();
}

//------------------------------------------------------------------------------
// Runtime → static dispatcher for tile sizes
//------------------------------------------------------------------------------
static cudaError_t
dispatch_tiles(const PlanCuTe& p,
               char ta, char tb,
               int m, int n, int k,
               float alpha,
               const float* A, int ldA,
               const float* B, int ldB,
               float  beta,
               float* C, int ldC,
               cudaStream_t stream)
{
    // Minimal initial support; extend with more cases as you add kernels.
    const bool NT = (p.path == PlanCuTe::Path::NT);
    const bool rA = (p.sA == SmemLayoutTag::ColMajor); // or your mapping
    const bool rB = (p.sB == SmemLayoutTag::ColMajor);
    const bool rC = (p.sC == SmemLayoutTag::ColMajor);
    if (p.blk_m == 128 && p.blk_n == 128 && p.blk_k == 8) {
      // Build a bitmask just to branch cleanly (bit0=rA, bit1=rB, bit2=rC, bit3=NT)
      const int mask = (rA ? 1 : 0) | (rB ? 2 : 0) | (rC ? 4 : 0) | (NT ? 8 : 0);

      switch (mask) {
          case 0:  // rA=0 rB=0 rC=0 NT=0
              return launch_for_tiles<128,128,8,false,false,false,false>(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 1:  // rA=1 rB=0 rC=0 NT=0
              return launch_for_tiles<128,128,8,true ,false,false,false>(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 2:  // rA=0 rB=1 rC=0 NT=0
              return launch_for_tiles<128,128,8,false,true ,false,false>(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 3:  // rA=1 rB=1 rC=0 NT=0
              return launch_for_tiles<128,128,8,true ,true ,false,false>(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 4:  // rA=0 rB=0 rC=1 NT=0
              return launch_for_tiles<128,128,8,false,false,true ,false>(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 5:  // rA=1 rB=0 rC=1 NT=0
              return launch_for_tiles<128,128,8,true ,false,true ,false>(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 6:  // rA=0 rB=1 rC=1 NT=0
              return launch_for_tiles<128,128,8,false,true ,true ,false>(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 7:  // rA=1 rB=1 rC=1 NT=0
              return launch_for_tiles<128,128,8,true ,true ,true ,false>(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 8:  // rA=0 rB=0 rC=0 NT=1
              return launch_for_tiles<128,128,8,false,false,false,true >(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 9:  // rA=1 rB=0 rC=0 NT=1
              return launch_for_tiles<128,128,8,true ,false,false,true >(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 10: // rA=0 rB=1 rC=0 NT=1
              return launch_for_tiles<128,128,8,false,true ,false,true >(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 11: // rA=1 rB=1 rC=0 NT=1
              return launch_for_tiles<128,128,8,true ,true ,false,true >(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 12: // rA=0 rB=0 rC=1 NT=1
              return launch_for_tiles<128,128,8,false,false,true ,true >(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 13: // rA=1 rB=0 rC=1 NT=1
              return launch_for_tiles<128,128,8,true ,false,true ,true >(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 14: // rA=0 rB=1 rC=1 NT=1
              return launch_for_tiles<128,128,8,false,true ,true ,true >(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);

          case 15: // rA=1 rB=1 rC=1 NT=1
              return launch_for_tiles<128,128,8,true ,true ,true ,true >(
                  p, ta, tb, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
      }
    }
    // Example extensions:
    // if (p.blk_m == 128 && p.blk_n == 64 && p.blk_k == 8)  return launch_for_tiles<128,64,8>(...);
    // if (p.blk_m == 64  && p.blk_n == 128 && p.blk_k == 8) return launch_for_tiles<64,128,8>(...);
    // if (p.blk_m == 64  && p.blk_n == 64  && p.blk_k == 8) return launch_for_tiles<64,64,8>(...);

    return cudaErrorNotSupported; // Unsupported tile for now
}

//------------------------------------------------------------------------------
// Public host launcher called by the adaptor (backend_adaptor.cpp)
//------------------------------------------------------------------------------
void gemm_launch(const PlanCuTe& plan,
                 char /*ta*/, char /*tb*/,
                 int m, int n, int k,
                 float alpha,
                 const float* A, int ldA,
                 const float* B, int ldB,
                 float  beta,
                 float* C, int ldC,
                 cudaStream_t stream)
{
    // We ignore ta/tb here and rely on plan.path (NT vs TN) for now.
    // If you want to cross-check, you can assert consistency.

    // Dispatch to a compile-time specialization for tiles
    (void)dispatch_tiles(plan, /*ta*/0, /*tb*/0,
                         m, n, k,
                         alpha, A, ldA, B, ldB, beta, C, ldC, stream);
}

} // namespace dphpc::backend::cute


// // Setup params for an NT GEMM
// // Use m-major smem sA, n-major smem sB, and mn-major threads tA|tB
// template <class TA, class TB, class TC,
//           class Alpha, class Beta>
// void
// gemm_nt(int m, int n, int k,
//         Alpha alpha,
//         TA const* A, int ldA,
//         TB const* B, int ldB,
//         Beta beta,
//         TC      * C, int ldC,
//         cudaStream_t stream = 0)
// {
//   using namespace cute;

//   // Define shapes (dynamic)
//   auto M = int(m);
//   auto N = int(n);
//   auto K = int(k);
//   auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

//   // Define NT strides (mixed)
//   auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
//   auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
//   auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

//   // Define CTA tile sizes (static)
//   auto bM = Int<128>{};
//   auto bN = Int<128>{};
//   auto bK = Int<  8>{};
//   auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

//   // Define the smem layouts (static)
//   auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
//   auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
//   auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

//   // Define the thread layouts (static)
//   auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (m,k) -> thr_idx
//   auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (n,k) -> thr_idx
//   auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx

//   dim3 dimBlock(size(tC));
//   dim3 dimGrid(size(ceil_div(M, bM)),
//                size(ceil_div(N, bN)));
//   gemm_device<<<dimGrid, dimBlock, 0, stream>>>
//       (prob_shape, cta_tiler,
//        A, dA, sA, tA,
//        B, dB, sB, tB,
//        C, dC, sC, tC,
//        alpha, beta);
// }

// // Setup params for a TN GEMM
// // Use padded m-major smem sA, padded n-major smem sB, and k-major threads tA|tB
// template <class TA, class TB, class TC,
//           class Alpha, class Beta>
// void
// gemm_tn(int m, int n, int k,
//         Alpha alpha,
//         TA const* A, int ldA,
//         TB const* B, int ldB,
//         Beta beta,
//         TC      * C, int ldC,
//         cudaStream_t stream = 0)
// {
//   using namespace cute;

//   // Define shapes (dynamic)
//   auto M = int(m);
//   auto N = int(n);
//   auto K = int(k);
//   auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

//   // Define TN strides (mixed)
//   auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
//   auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
//   auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

//   // Define CTA tile sizes (static)
//   auto bM = Int<128>{};
//   auto bN = Int<128>{};
//   auto bK = Int<  8>{};
//   auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

//   // Define the smem layouts (static)
//   auto sA = make_layout(make_shape(bM,bK), LayoutRight{});   // (m,k) -> smem_idx; k-major
//   auto sB = make_layout(make_shape(bN,bK), LayoutRight{});   // (n,k) -> smem_idx; k-major
//   auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

//   // Define the thread layouts (static)
//   auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (m,k) -> thr_idx; k-major
//   auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}), LayoutRight{});  // (n,k) -> thr_idx; k-major
//   auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));                 // (m,n) -> thr_idx; m-major

//   dim3 dimBlock(size(tC));
//   dim3 dimGrid(size(ceil_div(M, bM)),
//                size(ceil_div(N, bN)));
//   gemm_device<<<dimGrid, dimBlock, 0, stream>>>
//       (prob_shape, cta_tiler,
//        A, dA, sA, tA,
//        B, dB, sB, tB,
//        C, dC, sC, tC,
//        alpha, beta);
// }

// template <class TA, class TB, class TC,
//           class Alpha, class Beta>
// void
// gemm(char transA, char transB, int m, int n, int k,
//      Alpha alpha,
//      TA const* A, int ldA,
//      TB const* B, int ldB,
//      Beta beta,
//      TC      * C, int ldC,
//      cudaStream_t stream = 0)
// {
//   if (transA == 'N' && transB == 'T') {
//     return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
//   } else
//   if (transA == 'T' && transB == 'N') {
//     return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
//   }
//   assert(false && "Not implemented");
// }


// int main(int argc, char** argv)
// {
//   int m = 5120;
//   if (argc >= 2)
//     sscanf(argv[1], "%d", &m);

//   int n = 5120;
//   if (argc >= 3)
//     sscanf(argv[2], "%d", &n);

//   int k = 4096;
//   if (argc >= 4)
//     sscanf(argv[3], "%d", &k);

//   char transA = 'N';
//   if (argc >= 5)
//     sscanf(argv[4], "%c", &transA);

//   char transB = 'T';
//   if (argc >= 6)
//     sscanf(argv[5], "%c", &transB);

//   using TA = float;
//   using TB = float;
//   using TC = float;
//   using TI = float;

//   TI alpha = 1.0;
//   TI beta  = 0.0;

//   std::cout << "M = " << m << std::endl;
//   std::cout << "N = " << n << std::endl;
//   std::cout << "K = " << k << std::endl;
//   std::cout << "C = A^" << transA << " B^" << transB << std::endl;

//   cute::device_init(0);

//   thrust::host_vector<TA> h_A(m*k);
//   thrust::host_vector<TB> h_B(n*k);
//   thrust::host_vector<TC> h_C(m*n);

//   for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
//   for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
//   for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);

//   thrust::device_vector<TA> d_A = h_A;
//   thrust::device_vector<TB> d_B = h_B;
//   thrust::device_vector<TC> d_C = h_C;

//   double gflops = (2.0*m*n*k) * 1e-9;

//   const int timing_iterations = 100;
//   GPU_Clock timer;

//   int ldA = 0, ldB = 0, ldC = m;

//   if (transA == 'N') {
//     ldA = m;
//   } else if (transA == 'T') {
//     ldA = k;
//   } else {
//     assert(false);
//   }

//   if (transB == 'N') {
//     ldB = k;
//   } else if (transB == 'T') {
//     ldB = n;
//   } else {
//     assert(false);
//   }
//   // Run once
//   d_C = h_C;
//   gemm(transA, transB, m, n, k,
//        alpha,
//        d_A.data().get(), ldA,
//        d_B.data().get(), ldB,
//        beta,
//        d_C.data().get(), ldC);
//   CUTE_CHECK_LAST();
//   thrust::host_vector<TC> cute_result = d_C;

//   // Timing iterations
//   timer.start();
//   for (int i = 0; i < timing_iterations; ++i) {
//     gemm(transA, transB, m, n, k,
//          alpha,
//          d_A.data().get(), ldA,
//          d_B.data().get(), ldB,
//          beta,
//          d_C.data().get(), ldC);
//   }
//   double cute_time = timer.seconds() / timing_iterations;
//   CUTE_CHECK_LAST();
//   printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);
//   return 0;
// }