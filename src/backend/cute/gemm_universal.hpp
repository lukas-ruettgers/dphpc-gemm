#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/gemm/device/gemm_universal.h>  // ← switch to Universal
#include <cutlass/arch/arch.h>

// ================== your existing numeric-macro mapping ==================
// Types: DTYPE_* = {1=half, 2=float, 3=double(acc)}
// Layouts: LAYOUT_*_CODE = {0=row, 1=col}

#ifndef DTYPE_A
  #define DTYPE_A 1
#endif
#ifndef DTYPE_B
  #define DTYPE_B 1
#endif
#ifndef DTYPE_C
  #define DTYPE_C 2
#endif
#ifndef DTYPE_ACC
  #define DTYPE_ACC 2
#endif

#ifndef LAYOUT_A_CODE
  #define LAYOUT_A_CODE 0
#endif
#ifndef LAYOUT_B_CODE
  #define LAYOUT_B_CODE 1
#endif
#ifndef LAYOUT_C_CODE
  #define LAYOUT_C_CODE 0
#endif

#if   DTYPE_A == 1
  using TypeA = cutlass::half_t;
#elif DTYPE_A == 2
  using TypeA = float;
#else
  #error "Unsupported DTYPE_A"
#endif

#if   DTYPE_B == 1
  using TypeB = cutlass::half_t;
#elif DTYPE_B == 2
  using TypeB = float;
#else
  #error "Unsupported DTYPE_B"
#endif

#if   DTYPE_C == 1
  using TypeC = cutlass::half_t;
#elif DTYPE_C == 2
  using TypeC = float;
#else
  #error "Unsupported DTYPE_C"
#endif

#if   DTYPE_ACC == 2
  using TypeAcc = float;
#elif DTYPE_ACC == 3
  using TypeAcc = double;
#else
  #error "Unsupported DTYPE_ACC"
#endif

#if   LAYOUT_A_CODE == 0
  using LayoutA = cutlass::layout::RowMajor;
#else
  using LayoutA = cutlass::layout::ColumnMajor;
#endif

#if   LAYOUT_B_CODE == 0
  using LayoutB = cutlass::layout::RowMajor;
#else
  using LayoutB = cutlass::layout::ColumnMajor;
#endif

#if   LAYOUT_C_CODE == 0
  using LayoutC = cutlass::layout::RowMajor;
#else
  using LayoutC = cutlass::layout::ColumnMajor;
#endif

// (Optional) keep these for documentation; Universal doesn’t require them
#ifndef TB_M
  #define TB_M 128
#endif
#ifndef TB_N
  #define TB_N 128
#endif
#ifndef TB_K
  #define TB_K 32
#endif
#ifndef WP_M
  #define WP_M 64
#endif
#ifndef WP_N
  #define WP_N 64
#endif
#ifndef WP_K
  #define WP_K 32
#endif
#ifndef INST_M
  #define INST_M 16
#endif
#ifndef INST_N
  #define INST_N 8
#endif
#ifndef INST_K
  #define INST_K 16
#endif

// Alignment hints (still useful for input data layout sanity)
#ifndef ALIGN_A
  #define ALIGN_A 8
#endif
#ifndef ALIGN_B
  #define ALIGN_B 8
#endif

#ifndef STAGES
  #define STAGES 3
#endif

// Split-K control: Universal uses a runtime field, but we’ll feed a macro.
#ifndef SPLITK_SLICES
  #define SPLITK_SLICES 1   // 1 = serial; >1 = parallel Split-K
#endif

#ifndef ARCH_SM
  #define ARCH_SM 120
#endif

#ifndef USE_TF32
  #define USE_TF32 0
#endif
#if USE_TF32
  #ifndef CUTLASS_USE_TF32
    #define CUTLASS_USE_TF32 1
  #endif
#endif

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    TypeC, /*ElementsPerAccess*/ 1, TypeAcc, TypeC>;

// ================== Final GEMM alias (Universal) ==================
using GemmConfigured = cutlass::gemm::device::GemmUniversal<
    TypeA, LayoutA,
    TypeB, LayoutB,
    TypeC, LayoutC,
    TypeAcc>;
