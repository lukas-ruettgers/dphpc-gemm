/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/***************************************************************************************************
 Example contrasting the Stream-K parallel decomposition for GEMM threadblocks versus the
 "classic data-parallel" and "Split-K" decompositions.

 For more details regarding the Stream-K method, see "Stream-K: Work-centric Parallel Decomposition
 for Dense Matrix-Matrix Multiplication on the GPU" (https://arxiv.org/abs/2301.03598)

 Requires NVIDIA Ampere or newer device (SM80+).

 - To lock persistence mode, power (400W), clocks (1005MHz) for evaluation (assumes device 0 and A100)

     cutlass$ sudo nvidia-smi -pm 1 -i 0

     cutlass$ sudo nvidia-smi -i 0 -pl 400

     cutlass$ sudo nvidia-smi -i 0 -lgc 1005

 - Build and run:

     cutlass$ mkdir build

     cutlass$ cd build

     cutlass/build$ cmake .. -DCUTLASS_NVCC_ARCHS=80

     cutlass/build$ make 47_ampere_gemm_universal_streamk

     cutlass/build$ ./examples/47_ampere_gemm_universal_streamk/47_ampere_gemm_universal_streamk

        10000 timing iterations of 2048 x 2048 x 2048 matrix-matrix multiply

        Basic data-parallel GEMM
          Disposition: Passed
          Avg runtime: 0.112633 ms
          GFLOPs: 152530

        StreamK GEMM with default load-balancing
          Disposition: Passed
          Avg runtime: 0.0941929 ms
          GFLOPs: 182390
          Speedup vs Basic-DP: 1.196

        StreamK emulating basic data-parallel GEMM
          Disposition: Passed
          Avg runtime: 0.113119 ms
          GFLOPs: 151875
          Speedup vs Basic-DP: 0.996

        Basic split-K GEMM with tile-splitting factor 2
          Disposition: Passed
          Avg runtime: 0.104772 ms
          GFLOPs: 163973

        StreamK emulating Split-K GEMM with tile-splitting factor 2
          Disposition: Passed
          Avg runtime: 0.105379 ms
          GFLOPs: 163029
          Speedup vs Basic-SplitK: 0.994

 **************************************************************************************************/

#include <iostream>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk_dynamic.h"


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations (cutlass_tensorop_h16816gemm_128x128_32x4_nn_align8)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                      // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C/D matrices in units of elements (up to 16 bytes)

// Multiply-accumulate blocking/pipelining details
using ElementAccumulator  = cutlass::half_t;                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm80;                      // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;           // Operator class tag
#ifdef TB160
using ThreadblockShape    = cutlass::gemm::GemmShape<160, 160, 32>;   // Threadblock-level tile size (concept: GemmShape)
using WarpShape           = cutlass::gemm::GemmShape<32, 32, 32>;     // Warp-level tile size (concept: GemmShape)
#elif defined(TB192) 
using ThreadblockShape    = cutlass::gemm::GemmShape<192, 160, 96>;   // Threadblock-level tile size (concept: GemmShape)
using WarpShape           = cutlass::gemm::GemmShape<32, 32, 96>;     // Warp-level tile size (concept: GemmShape)
#else
using ThreadblockShape    = cutlass::gemm::GemmShape<128, 128, 32>;   // Threadblock-level tile size (concept: GemmShape)
using WarpShape           = cutlass::gemm::GemmShape<32, 32, 32>;     // Warp-level tile size (concept: GemmShape)
#endif
using ThreadblockShape32  = cutlass::gemm::GemmShape<32, 32, 32>;   // Threadblock-level tile size (concept: GemmShape)
using ThreadblockShape160  = cutlass::gemm::GemmShape<160, 160, 32>;   // Threadblock-level tile size (concept: GemmShape)
using WarpShape160         = cutlass::gemm::GemmShape<32, 32, 32>;     // Warp-level tile size (concept: GemmShape)
using ThreadblockShape192  = cutlass::gemm::GemmShape<192, 192, 96>;   // Threadblock-level tile size (concept: GemmShape)
using WarpShape192         = cutlass::gemm::GemmShape<32, 32, 96>;     // Warp-level tile size (concept: GemmShape)
using InstructionShape    = cutlass::gemm::GemmShape<16, 8, 16>;      // Instruction-level tile size (concept: GemmShape)
constexpr int NumStages   = 4;                                        // Number of global->shared pipeline stages used in the GEMM mainloop

// Epilogue output operator
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,               // Element type for C and D matrix operands
    AlignmentC,             // Memory access granularity of C and D matrix in units of elements
    ElementAccumulator,     // Element type from internal accumaccumulation
    ElementAccumulator>;    // Data type used to compute linear combination

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

// Classic data-parallel device GEMM implementation type
using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB>;

using DeviceGemmBasicCompare = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape160,
    WarpShape160,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB>;

using DeviceGemmBasicCompare192 = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape192,
    WarpShape192,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB>;
// StreamK device GEMM implementation type
template<int CtaShapeM, int CtaShapeN>
using DeviceGemmStreamK = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK<CtaShapeM,CtaShapeN>, // <-- Only difference
    NumStages,
    AlignmentA,
    AlignmentB>;

    // StreamK device GEMM implementation type
template<int CtaShapeM, int CtaShapeN>
using DeviceGemmStreamK160 = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape160,
    WarpShape160,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK<CtaShapeM,CtaShapeN>, // <-- Only difference
    NumStages,
    AlignmentA,
    AlignmentB>;

// StreamK device GEMM implementation type
template<int CtaShapeM, int CtaShapeN>
using DeviceGemmStreamKDyn = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamKDyn<CtaShapeM,CtaShapeN,false,false>, // <-- Only difference
    NumStages,
    AlignmentA,
    AlignmentB>;

using StreamKDyn0 = DeviceGemmStreamKDyn<1,1>;
using StreamKDyn1 = DeviceGemmStreamKDyn<2,1>;
using StreamKDyn2 = DeviceGemmStreamKDyn<4,1>;
using StreamKDyn3 = DeviceGemmStreamKDyn<8,1>;
using StreamKDyn4 = DeviceGemmStreamKDyn<16,1>;
using StreamKDyn5 = DeviceGemmStreamKDyn<32,1>;
using StreamKDyn6 = DeviceGemmStreamKDyn<64,1>;
using StreamKDyn7 = DeviceGemmStreamKDyn<128,1>;
using StreamKDyn8 = DeviceGemmStreamKDyn<256,1>;
using StreamKDyn9 = DeviceGemmStreamKDyn<512,1>;
using StreamKDyn10 = DeviceGemmStreamKDyn<1024,1>;
using StreamKDyn11 = DeviceGemmStreamKDyn<2048,1>;
using StreamKDyn12 = DeviceGemmStreamKDyn<4096,1>;
using StreamKDyn13 = DeviceGemmStreamKDyn<8192,1>;
using StreamKDyn14 = DeviceGemmStreamKDyn<16384,1>;
using StreamKDyn15 = DeviceGemmStreamKDyn<32768,1>;
using StreamKDyn16 = DeviceGemmStreamKDyn<65536,1>;
using StreamKDyn17 = DeviceGemmStreamKDyn<131072,1>;
using StreamKDyn18 = DeviceGemmStreamKDyn<262144,1>;
using StreamKDyn19 = DeviceGemmStreamKDyn<524288,1>;

using DeviceGemmStreamKDynBaseline = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamKDyn<8,4,true, false>, // <-- Only difference
    NumStages,
    AlignmentA,
    AlignmentB>;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(true)
  {}

};


/// Command line options parsing
struct Options
{
  std::string               command_name;
  bool                      help;
  cutlass::gemm::GemmCoord  problem_size;
  float                     alpha;
  float                     beta;
  int                       split_k_factor;
  int                       avail_sms;
  bool                      reference_check;
  int                       iterations;
  int                       iterations_major;
  bool                      dp;
  bool                      sk;
  bool                      init_dyn_only;
  bool                      streamk;
  bool                      baseline;
  bool                      run;
  int                       cohort_m;
  int                       cohort_n;

  cutlass::HostTensor<ElementA, LayoutA> tensor_a;
  cutlass::HostTensor<ElementB, LayoutB> tensor_b;
  cutlass::HostTensor<ElementC, LayoutC> tensor_c;
  cutlass::HostTensor<ElementC, LayoutC> tensor_d;
  cutlass::HostTensor<ElementC, LayoutC> tensor_ref_d;

  Options(std::string command_name) :
    command_name(command_name),
    help(false),
    problem_size({2048, 2048, 2048}),
    alpha(1.0f),
    beta(0.0f),
    split_k_factor(1),
    avail_sms(-1),              // Number of device SMs to use is unlimited
    reference_check(true),
    iterations(10000),
    iterations_major(1),
    sk(false),
    dp(false),
    init_dyn_only(false),
    streamk(false),
    baseline(false),
    run(false),
    cohort_m(8),
    cohort_n(4)
  {}

  bool valid() const
  {
    return true;
  }

  void parse(int argc, char const **args)
  {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("split", split_k_factor);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("iterations_major", iterations_major);
    cmd.get_cmd_line_argument("dp", dp);
    cmd.get_cmd_line_argument("sk", sk);
    cmd.get_cmd_line_argument("init_dyn_only", init_dyn_only);
    cmd.get_cmd_line_argument("baseline", baseline);
    cmd.get_cmd_line_argument("streamk", streamk);
    cmd.get_cmd_line_argument("run", run);
    cmd.get_cmd_line_argument("cohort_m", cohort_m);
    cmd.get_cmd_line_argument("cohort_n", cohort_n);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const
  {
    out
      << "Performs a GEMM computation.\n"
      << "\n"
      << "Options:\n"
      << "\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --split=<int>               Split-K factor to emulate\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << command_name << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    return 2.0 * double(problem_size.product()) / double(1.0e9) / runtime_s;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Populates a DeviceGemmBasic::Arguments structure from the given commandline options
template <typename DeviceGemmT>
typename DeviceGemmT::Arguments args_from_options(
    const DeviceGemmT &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA, LayoutA> &tensor_a,
    cutlass::HostTensor<ElementB, LayoutB> &tensor_b,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_c,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_d)
{
  return typename DeviceGemmT::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
    options.problem_size,                     // problem_size
    options.split_k_factor,                   // batch count / splitk slices
    {                                         // epilogue parameters
      ElementAccumulator(options.alpha),
      ElementAccumulator(options.beta)
    },
    tensor_a.device_data(),                   // ptr_A
    tensor_b.device_data(),                   // ptr_B
    tensor_c.device_data(),                   // ptr_C
    tensor_d.device_data(),                   // ptr_D
    options.problem_size.mk().product(),      // batch_stride_A
    options.problem_size.nk().product(),      // batch_stride_B
    options.problem_size.mn().product(),      // batch_stride_C
    options.problem_size.mn().product(),      // batch_stride_D
    tensor_a.layout().stride(0),              // stride_a
    tensor_b.layout().stride(0),              // stride_b
    tensor_c.layout().stride(0),              // stride_c
    tensor_d.layout().stride(0));             // stride_d
}

/// Populates a DeviceGemmStreamK::Arguments structure from the given commandline options
template <int CohortSizeM, int CohortSizeN>
typename DeviceGemmStreamK<CohortSizeM, CohortSizeN>::Arguments args_from_options(
    const DeviceGemmStreamK<CohortSizeM, CohortSizeN> &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA, LayoutA> &tensor_a,
    cutlass::HostTensor<ElementB, LayoutB> &tensor_b,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_c,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_d)
{
  return typename DeviceGemmStreamK<CohortSizeM, CohortSizeN>::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
    options.problem_size,                     // problem_size
    options.split_k_factor,                   // batch count / splitk slices
    {                                         // epilogue parameters
      ElementAccumulator(options.alpha),
      ElementAccumulator(options.beta)
    },
    tensor_a.device_data(),                   // ptr_A
    tensor_b.device_data(),                   // ptr_B
    tensor_c.device_data(),                   // ptr_C
    tensor_d.device_data(),                   // ptr_D
    options.problem_size.mk().product(),      // batch_stride_A
    options.problem_size.nk().product(),      // batch_stride_B
    options.problem_size.mn().product(),      // batch_stride_C
    options.problem_size.mn().product(),      // batch_stride_D
    tensor_a.layout().stride(0),              // stride_a
    tensor_b.layout().stride(0),              // stride_b
    tensor_c.layout().stride(0),              // stride_c
    tensor_d.layout().stride(0),              // stride_d
    options.avail_sms);                       // avail_sms
}

/// Populates a DeviceGemmStreamK::Arguments structure from the given commandline options
template <int CohortSizeM, int CohortSizeN>
typename DeviceGemmStreamK160<CohortSizeM, CohortSizeN>::Arguments args_from_options(
    const DeviceGemmStreamK160<CohortSizeM, CohortSizeN> &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA, LayoutA> &tensor_a,
    cutlass::HostTensor<ElementB, LayoutB> &tensor_b,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_c,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_d)
{
  return typename DeviceGemmStreamK160<CohortSizeM, CohortSizeN>::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
    options.problem_size,                     // problem_size
    options.split_k_factor,                   // batch count / splitk slices
    {                                         // epilogue parameters
      ElementAccumulator(options.alpha),
      ElementAccumulator(options.beta)
    },
    tensor_a.device_data(),                   // ptr_A
    tensor_b.device_data(),                   // ptr_B
    tensor_c.device_data(),                   // ptr_C
    tensor_d.device_data(),                   // ptr_D
    options.problem_size.mk().product(),      // batch_stride_A
    options.problem_size.nk().product(),      // batch_stride_B
    options.problem_size.mn().product(),      // batch_stride_C
    options.problem_size.mn().product(),      // batch_stride_D
    tensor_a.layout().stride(0),              // stride_a
    tensor_b.layout().stride(0),              // stride_b
    tensor_c.layout().stride(0),              // stride_c
    tensor_d.layout().stride(0),              // stride_d
    options.avail_sms);                       // avail_sms
}
/// Populates a DeviceGemmStreamKDyn::Arguments structure from the given commandline options
template <int CohortSizeM, int CohortSizeN>
typename DeviceGemmStreamKDyn<CohortSizeM, CohortSizeN>::Arguments args_from_options(
    const DeviceGemmStreamKDyn<CohortSizeM, CohortSizeN> &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA, LayoutA> &tensor_a,
    cutlass::HostTensor<ElementB, LayoutB> &tensor_b,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_c,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_d)
{
  return typename DeviceGemmStreamKDyn<CohortSizeM, CohortSizeN>::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
    options.problem_size,                     // problem_size
    options.split_k_factor,                   // batch count / splitk slices
    {                                         // epilogue parameters
      ElementAccumulator(options.alpha),
      ElementAccumulator(options.beta)
    },
    tensor_a.device_data(),                   // ptr_A
    tensor_b.device_data(),                   // ptr_B
    tensor_c.device_data(),                   // ptr_C
    tensor_d.device_data(),                   // ptr_D
    options.problem_size.mk().product(),      // batch_stride_A
    options.problem_size.nk().product(),      // batch_stride_B
    options.problem_size.mn().product(),      // batch_stride_C
    options.problem_size.mn().product(),      // batch_stride_D
    tensor_a.layout().stride(0),              // stride_a
    tensor_b.layout().stride(0),              // stride_b
    tensor_c.layout().stride(0),              // stride_c
    tensor_d.layout().stride(0),              // stride_d
    options.avail_sms);                       // avail_sms
}

template <typename DeviceGemmT>
Result check(std::string description, Options &options)
{
  // Display test description
  std::cout << std::endl << description << std::endl;

  // Zero-initialize test output matrix D
  cutlass::reference::host::TensorFill(options.tensor_d.host_view());
  options.tensor_d.sync_device();

  // Instantiate CUTLASS kernel depending on templates
  DeviceGemmT device_gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmT
  auto arguments = args_from_options(device_gemm, options, options.tensor_a, options.tensor_b, options.tensor_c, options.tensor_d);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = DeviceGemmT::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  CUTLASS_CHECK(device_gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(device_gemm());

  // Copy output data from CUTLASS and reference kernel to host for comparison
  options.tensor_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = cutlass::reference::host::TensorEquals(
    options.tensor_d.host_view(),
    options.tensor_ref_d.host_view());

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
  return result;
}

/// Populates a DeviceGemmStreamKDyn::Arguments structure from the given commandline options
typename DeviceGemmStreamKDynBaseline::Arguments args_from_options(
    const DeviceGemmStreamKDynBaseline &device_gemm,
    const Options &options,
    cutlass::HostTensor<ElementA, LayoutA> &tensor_a,
    cutlass::HostTensor<ElementB, LayoutB> &tensor_b,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_c,
    cutlass::HostTensor<ElementC, LayoutC> &tensor_d)
{
  return typename DeviceGemmStreamKDynBaseline::Arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
    options.problem_size,                     // problem_size
    options.split_k_factor,                   // batch count / splitk slices
    {                                         // epilogue parameters
      ElementAccumulator(options.alpha),
      ElementAccumulator(options.beta)
    },
    tensor_a.device_data(),                   // ptr_A
    tensor_b.device_data(),                   // ptr_B
    tensor_c.device_data(),                   // ptr_C
    tensor_d.device_data(),                   // ptr_D
    options.problem_size.mk().product(),      // batch_stride_A
    options.problem_size.nk().product(),      // batch_stride_B
    options.problem_size.mn().product(),      // batch_stride_C
    options.problem_size.mn().product(),      // batch_stride_D
    tensor_a.layout().stride(0),              // stride_a
    tensor_b.layout().stride(0),              // stride_b
    tensor_c.layout().stride(0),              // stride_c
    tensor_d.layout().stride(0),              // stride_d
    options.avail_sms);                       // avail_sms
}

/// Execute a given example GEMM computation
template <typename DeviceGemmT>
Result run(std::string description, Options &options)
{
  // Display test description
  std::cout << std::endl << description << std::endl;

  // Zero-initialize test output matrix D
  cutlass::reference::host::TensorFill(options.tensor_d.host_view());
  options.tensor_d.sync_device();

  // Instantiate CUTLASS kernel depending on templates
  DeviceGemmT device_gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmT
  auto arguments = args_from_options(device_gemm, options, options.tensor_a, options.tensor_b, options.tensor_c, options.tensor_d);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = DeviceGemmT::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  
  // Check the problem size is supported or not
  std::cout << "Can implement..." << std::endl;
  CUTLASS_CHECK(device_gemm.can_implement(arguments));
  
  // Initialize CUTLASS kernel with arguments and workspace pointer
  std::cout << "Initialize..." << std::endl;
  CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));
  // Correctness / Warmup iteration
  std::cout << "Start run" << std::endl;
  CUTLASS_CHECK(device_gemm());
  std::cout << "End   run" << std::endl;

  // Copy output data from CUTLASS and reference kernel to host for comparison
  options.tensor_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = cutlass::reference::host::TensorEquals(
    options.tensor_d.host_view(),
    options.tensor_ref_d.host_view());

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  // Run profiling loop
  std::vector<double> times_ms;
  times_ms.reserve(options.iterations);

  if (options.iterations_major > 0){
    for(int iter_major=0; iter_major < options.iterations_major; iter_major++){
      if (options.iterations > 0)
      {
        GpuTimer timer;
        timer.start();
        for (int iter = 0; iter < options.iterations; ++iter) {
          CUTLASS_CHECK(device_gemm());
        }
        timer.stop();
    
        // Compute average runtime and GFLOPs.
        float elapsed_ms = timer.elapsed_millis();
        result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
        result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);
    
        std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
        std::cout << "  GFLOPs: " << result.gflops << std::endl;
      }
    }
  }

  if (!result.passed) {
    exit(-1);
  }

  return result;
}

template <template<int,int> class DeviceGemmStreamK, int CohortSizeM, int CohortSizeN>
Result run(std::string description, Options &options)
{
  // Display test description
  std::cout << std::endl << description << std::endl;

  // Zero-initialize test output matrix D
  cutlass::reference::host::TensorFill(options.tensor_d.host_view());
  options.tensor_d.sync_device();

  // Instantiate CUTLASS kernel depending on templates
  DeviceGemmStreamKDyn<CohortSizeM, CohortSizeN> device_gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmT
  auto arguments = args_from_options(device_gemm, options, options.tensor_a, options.tensor_b, options.tensor_c, options.tensor_d);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = DeviceGemmStreamKDyn<CohortSizeM, CohortSizeN>::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  CUTLASS_CHECK(device_gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(device_gemm());

  // Copy output data from CUTLASS and reference kernel to host for comparison
  options.tensor_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = cutlass::reference::host::TensorEquals(
    options.tensor_d.host_view(),
    options.tensor_ref_d.host_view());

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(device_gemm());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPs: " << result.gflops << std::endl;
  }

  if (!result.passed) {
    exit(-1);
  }

  return result;
}

/*
template <template<int,int> class DeviceGemmStreamKDyn, int CohortSizeM, int CohortSizeN>
Result run(std::string description, Options &options)
{
  // Display test description
  std::cout << std::endl << description << std::endl;

  // Zero-initialize test output matrix D
  cutlass::reference::host::TensorFill(options.tensor_d.host_view());
  options.tensor_d.sync_device();

  // Instantiate CUTLASS kernel depending on templates
  DeviceGemmStreamKDyn<CohortSizeM, CohortSizeN> device_gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmT
  auto arguments = args_from_options(device_gemm, options, options.tensor_a, options.tensor_b, options.tensor_c, options.tensor_d);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = DeviceGemmStreamKDyn<CohortSizeM, CohortSizeN>::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  CUTLASS_CHECK(device_gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(device_gemm());

  // Copy output data from CUTLASS and reference kernel to host for comparison
  options.tensor_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = cutlass::reference::host::TensorEquals(
    options.tensor_d.host_view(),
    options.tensor_ref_d.host_view());

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(device_gemm());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPs: " << result.gflops << std::endl;
  }

  if (!result.passed) {
    exit(-1);
  }

  return result;
}
*/

/// Program entrypoint
int main(int argc, const char **argv)
{
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;

    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  // Current device must must have compute capability at least 80
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  if (!((props.major * 10 + props.minor) >= 80))
  {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;

    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  // Parse commandline options
  Options options("ampere_streamk_gemm");
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  std::cout <<
    options.iterations << " timing iterations of " <<
    options.problem_size.m() << " x " <<
    options.problem_size.n() << " x " <<
    options.problem_size.k() << " matrix-matrix multiply" << std::endl;

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }


  //
  // Initialize GEMM datasets
  //

  // Initialize tensors using CUTLASS helper functions
  options.tensor_a.resize(options.problem_size.mk());       // <- Create matrix A with dimensions M x K
  options.tensor_b.resize(options.problem_size.kn());       // <- Create matrix B with dimensions K x N
  options.tensor_c.resize(options.problem_size.mn());       // <- Create matrix C with dimensions M x N
  options.tensor_d.resize(options.problem_size.mn());       // <- Create matrix D with dimensions M x N used to store output from CUTLASS kernel
  options.tensor_ref_d.resize(options.problem_size.mn());   // <- Create matrix D with dimensions M x N used to store output from reference kernel

  // Fill matrix A on host with uniform-random data [-2, 2]
  cutlass::reference::host::TensorFillRandomUniform(
      options.tensor_a.host_view(),
      1,
      ElementA(2),
      ElementA(-2),
      0);

  // Fill matrix B on host with uniform-random data [-2, 2]
  cutlass::reference::host::TensorFillRandomUniform(
      options.tensor_b.host_view(),
      1,
      ElementB(2),
      ElementB(-2),
      0);

  // Fill matrix C on host with uniform-random data [-2, 2]
  cutlass::reference::host::TensorFillRandomUniform(
      options.tensor_c.host_view(),
      1,
      ElementC(2),
      ElementC(-2),
      0);


  //
  // Compute reference output
  //

  // Copy data from host to GPU
  options.tensor_a.sync_device();
  options.tensor_b.sync_device();
  options.tensor_c.sync_device();

  // Zero-initialize reference output matrix D
  cutlass::reference::host::TensorFill(options.tensor_ref_d.host_view());
  options.tensor_ref_d.sync_device();

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
    options.problem_size,
    ElementAccumulator(options.alpha),
    options.tensor_a.device_ref(),
    options.tensor_b.device_ref(),
    ElementAccumulator(options.beta),
    options.tensor_c.device_ref(),
    options.tensor_ref_d.device_ref());

  // Wait for kernels to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy output data from reference kernel to host for comparison
  options.tensor_ref_d.sync_host();


  //
  // Evaluate CUTLASS kernels
  //
  

  if (options.init_dyn_only){
    
    if(options.cohort_n == 1){
    switch (options.cohort_m) {
    case 0:  run<DeviceGemmStreamKDyn<(1<<0),1 >>("Extended StreamK GEMM, C_M={0}", options); break;
    case 1:  run<DeviceGemmStreamKDyn<(1<<1),1 >>("Extended StreamK GEMM, C_M={1}", options); break;
    case 2:  run<DeviceGemmStreamKDyn<(1<<2),1 >>("Extended StreamK GEMM, C_M={2}", options); break;
    case 3:  run<DeviceGemmStreamKDyn<(1<<3),1 >>("Extended StreamK GEMM, C_M={3}", options); break;
    case 4:  run<DeviceGemmStreamKDyn<(1<<4),1 >>("Extended StreamK GEMM, C_M={4}", options); break;
    case 5:  run<DeviceGemmStreamKDyn<(1<<5),1 >>("Extended StreamK GEMM, C_M={5}", options); break;
    case 6:  run<DeviceGemmStreamKDyn<(1<<6),1 >>("Extended StreamK GEMM, C_M={6}", options); break;
    case 7:  run<DeviceGemmStreamKDyn<(1<<7),1 >>("Extended StreamK GEMM, C_M={7}", options); break;
    case 8:  run<DeviceGemmStreamKDyn<(1<<8),1 >>("Extended StreamK GEMM, C_M={8}", options); break;
    case 9:  run<DeviceGemmStreamKDyn<(1<<9),1 >>("Extended StreamK GEMM, C_M={9}", options); break;
    case 10: run<DeviceGemmStreamKDyn<(1<<10),1>>("Extended StreamK GEMM, C_M={10}", options); break;
    case 11: run<DeviceGemmStreamKDyn<(1<<11),1>>("Extended StreamK GEMM, C_M={11}", options); break;
    case 12: run<DeviceGemmStreamKDyn<(1<<12),1>>("Extended StreamK GEMM, C_M={12}", options); break;
    case 13: run<DeviceGemmStreamKDyn<(1<<13),1>>("Extended StreamK GEMM, C_M={13}", options); break;
    case 14: run<DeviceGemmStreamKDyn<(1<<14),1>>("Extended StreamK GEMM, C_M={14}", options); break;
    case 15: run<DeviceGemmStreamKDyn<(1<<15),1>>("Extended StreamK GEMM, C_M={15}", options); break;
    case 16: run<DeviceGemmStreamKDyn<(1<<16),1>>("Extended StreamK GEMM, C_M={16}", options); break;
    case 17: run<DeviceGemmStreamKDyn<(1<<17),1>>("Extended StreamK GEMM, C_M={17}", options); break;
    case 18: run<DeviceGemmStreamKDyn<(1<<18),1>>("Extended StreamK GEMM, C_M={18}", options); break;
    case 19: run<DeviceGemmStreamKDyn<(1<<19),1>>("Extended StreamK GEMM, C_M={19}", options); break;
  
    default:
      throw std::runtime_error("Unsupported cohort_m value");
    }
  }
  else if(options.cohort_m==1){
    switch (options.cohort_n) {
    case 0:  run<DeviceGemmStreamKDyn<1,(1<<0) >>("Extended StreamK GEMM, C_N={0}", options); break;
    case 1:  run<DeviceGemmStreamKDyn<1,(1<<1) >>("Extended StreamK GEMM, C_N={1}", options); break;
    case 2:  run<DeviceGemmStreamKDyn<1,(1<<2) >>("Extended StreamK GEMM, C_N={2}", options); break;
    case 3:  run<DeviceGemmStreamKDyn<1,(1<<3) >>("Extended StreamK GEMM, C_N={3}", options); break;
    case 4:  run<DeviceGemmStreamKDyn<1,(1<<4) >>("Extended StreamK GEMM, C_N={4}", options); break;
    case 5:  run<DeviceGemmStreamKDyn<1,(1<<5) >>("Extended StreamK GEMM, C_N={5}", options); break;
    case 6:  run<DeviceGemmStreamKDyn<1,(1<<6) >>("Extended StreamK GEMM, C_N={6}", options); break;
    case 7:  run<DeviceGemmStreamKDyn<1,(1<<7) >>("Extended StreamK GEMM, C_N={7}", options); break;
    case 8:  run<DeviceGemmStreamKDyn<1,(1<<8) >>("Extended StreamK GEMM, C_N={8}", options); break;
    case 9:  run<DeviceGemmStreamKDyn<1,(1<<9) >>("Extended StreamK GEMM, C_N={9}", options); break;
    case 10: run<DeviceGemmStreamKDyn<1,(1<<10)>>("Extended StreamK GEMM, C_N={10}", options); break;
    case 11: run<DeviceGemmStreamKDyn<1,(1<<11)>>("Extended StreamK GEMM, C_N={11}", options); break;
    case 12: run<DeviceGemmStreamKDyn<1,(1<<12)>>("Extended StreamK GEMM, C_N={12}", options); break;
    case 13: run<DeviceGemmStreamKDyn<1,(1<<13)>>("Extended StreamK GEMM, C_N={13}", options); break;
    case 14: run<DeviceGemmStreamKDyn<1,(1<<14)>>("Extended StreamK GEMM, C_N={14}", options); break;
    case 15: run<DeviceGemmStreamKDyn<1,(1<<15)>>("Extended StreamK GEMM, C_N={15}", options); break;
    case 16: run<DeviceGemmStreamKDyn<1,(1<<16)>>("Extended StreamK GEMM, C_N={16}", options); break;
    case 17: run<DeviceGemmStreamKDyn<1,(1<<17)>>("Extended StreamK GEMM, C_N={17}", options); break;
    case 18: run<DeviceGemmStreamKDyn<1,(1<<18)>>("Extended StreamK GEMM, C_N={18}", options); break;
    case 19: run<DeviceGemmStreamKDyn<1,(1<<19)>>("Extended StreamK GEMM, C_N={19}", options); break;
  
    default:
      throw std::runtime_error("Unsupported cohort_n value");
    }
    
  }
  else if(options.cohort_m==8 && options.cohort_n==4){
    run<DeviceGemmStreamKDyn<8,4>>("Extended StreamK GEMM (8,4)", options);
  }
  else if(options.cohort_m==4 && options.cohort_n==8){
    run<DeviceGemmStreamKDyn<4,8>>("Extended StreamK GEMM (4,8)", options);
  }
  else if(options.cohort_m==6 && options.cohort_n==6){
    run<DeviceGemmStreamKDyn<6,6>>("Extended StreamK GEMM (6,6)", options);
  }
    return 0;
  }
  if(options.streamk){
    if(options.cohort_m == 6 && options.cohort_n==6){
      Result streamk_default  = run<DeviceGemmStreamK<6,6>>("StreamK GEMM with default load-balancing (6,6)", options);
    }
    else if(options.cohort_m==4 && options.cohort_n==8){
      run<DeviceGemmStreamK<4,8>>("StreamK GEMM (4,8)", options);
    }
    else if(options.cohort_n==1){
      switch (options.cohort_m) {
      case 0:  run<DeviceGemmStreamK<(1<<0),1 >>("StreamK GEMM, C_M={0}", options); break;
      case 1:  run<DeviceGemmStreamK<(1<<1),1 >>("StreamK GEMM, C_M={1}", options); break;
      case 2:  run<DeviceGemmStreamK<(1<<2),1 >>("StreamK GEMM, C_M={2}", options); break;
      case 3:  run<DeviceGemmStreamK<(1<<3),1 >>("StreamK GEMM, C_M={3}", options); break;
      case 4:  run<DeviceGemmStreamK<(1<<4),1 >>("StreamK GEMM, C_M={4}", options); break;
      case 5:  run<DeviceGemmStreamK<(1<<5),1 >>("StreamK GEMM, C_M={5}", options); break;
      case 6:  run<DeviceGemmStreamK<(1<<6),1 >>("StreamK GEMM, C_M={6}", options); break;
      case 7:  run<DeviceGemmStreamK<(1<<7),1 >>("StreamK GEMM, C_M={7}", options); break;
      case 8:  run<DeviceGemmStreamK<(1<<8),1 >>("StreamK GEMM, C_M={8}", options); break;
      case 9:  run<DeviceGemmStreamK<(1<<9),1 >>("StreamK GEMM, C_M={9}", options); break;
      case 10: run<DeviceGemmStreamK<(1<<10),1>>("StreamK GEMM, C_M={10}", options); break;
      case 11: run<DeviceGemmStreamK<(1<<11),1>>("StreamK GEMM, C_M={11}", options); break;
      case 12: run<DeviceGemmStreamK<(1<<12),1>>("StreamK GEMM, C_M={12}", options); break;
      case 13: run<DeviceGemmStreamK<(1<<13),1>>("StreamK GEMM, C_M={13}", options); break;
      case 14: run<DeviceGemmStreamK<(1<<14),1>>("StreamK GEMM, C_M={14}", options); break;
      case 15: run<DeviceGemmStreamK<(1<<15),1>>("StreamK GEMM, C_M={15}", options); break;
      case 16: run<DeviceGemmStreamK<(1<<16),1>>("StreamK GEMM, C_M={16}", options); break;
      case 17: run<DeviceGemmStreamK<(1<<17),1>>("StreamK GEMM, C_M={17}", options); break;
      case 18: run<DeviceGemmStreamK<(1<<18),1>>("StreamK GEMM, C_M={18}", options); break;
      case 19: run<DeviceGemmStreamK<(1<<19),1>>("StreamK GEMM, C_M={19}", options); break;
    
      default:
        throw std::runtime_error("Unsupported cohort_m value");
      }
    }
    else{
      Result streamk_default  = run<DeviceGemmStreamK<8,4>>("StreamK GEMM with default load-balancing (8,4)", options);
    }
    return 0;
  }
  if(options.baseline){
    run<DeviceGemmBasic>("Basic data-parallel GEMM", options);
    return 0;
  }
  
  // Test default operation
  if (options.run == 1)
  {
    // Compare basic data-parallel version versus StreamK version using default load-balancing heuristics
    if(options.dp){
      Result basic_dp         = run<DeviceGemmBasic>("Basic data-parallel GEMM", options);
      // Result basic_dp_compare192 = run<DeviceGemmBasicCompare192>("Adjusted data-parallel GEMM, 192", options);
      Result basic_dp_compare160 = run<DeviceGemmBasicCompare>("Adjusted data-parallel GEMM, 160", options);
      printf("  Speedup Basic-DP 160 vs Basic-DP: %.3f\n", (basic_dp.avg_runtime_ms / basic_dp_compare160.avg_runtime_ms));
      return 0;
    }
    if(options.sk){
      Result streamk_default  = run<DeviceGemmStreamK<8,4>>("StreamK GEMM with default load-balancing", options);
      Result streamk_default160= run<DeviceGemmStreamK160<8,4>>("StreamK GEMM with default load-balancing", options);
      printf("  Speedup Stream-K 160 vs Stream-K: %.3f\n", (streamk_default.avg_runtime_ms / streamk_default160.avg_runtime_ms));
      return 0;
    }
      Result basic_dp         = run<DeviceGemmBasic>("Basic data-parallel GEMM", options);
      // Result basic_dp_compare192 = run<DeviceGemmBasicCompare192>("Adjusted data-parallel GEMM, 192", options);
      Result basic_dp_compare160 = run<DeviceGemmBasicCompare>("Adjusted data-parallel GEMM, 160", options);
    Result streamk_default  = run<DeviceGemmStreamK<8,4>>("StreamK GEMM with default load-balancing", options);
      Result streamk_default160= run<DeviceGemmStreamK160<8,4>>("StreamK GEMM with default load-balancing", options);
    Result streamk_default66  = run<DeviceGemmStreamK<6,6>>("StreamK GEMM with (6,6) load-balancing", options);
    Result streamk_dyn_base = run<DeviceGemmStreamKDynBaseline>("Extended StreamK GEMM emulating classic StreamK", options);
    Result streamk_dyn      = run<DeviceGemmStreamKDyn<8,4>>("Extended StreamK GEMM", options);

    // printf("  Speedup Basic-DP 192 vs Basic-DP: %.3f\n", (basic_dp.avg_runtime_ms / basic_dp_compare192.avg_runtime_ms));
    printf("  Speedup Basic-DP 160 vs Basic-DP: %.3f\n", (basic_dp.avg_runtime_ms / basic_dp_compare160.avg_runtime_ms));
    printf("  Speedup Stream-K vs Basic-DP: %.3f\n", (basic_dp.avg_runtime_ms / streamk_default.avg_runtime_ms));
    printf("  Speedup Stream-K 6,6 vs Stream-K: %.3f\n", (streamk_default.avg_runtime_ms / streamk_default66.avg_runtime_ms));
    printf("  Speedup Stream-K 160 vs Stream-K: %.3f\n", (streamk_default.avg_runtime_ms / streamk_default160.avg_runtime_ms));
    printf("  Speedup Stream-K Dyn vs Stream-K: %.3f\n", (streamk_default.avg_runtime_ms / streamk_dyn.avg_runtime_ms));
    printf("  Speedup Stream-K Dyn vs Emulating Stream-K: %.3f\n", (streamk_dyn_base.avg_runtime_ms / streamk_dyn.avg_runtime_ms));

    return 0;
    /*
    // Show that StreamK can emulate basic data-parallel GEMM when we set the number of SMs to load-balance across = 1
    options.avail_sms       = 1;        // Set loadbalancing width to 1 SM (no load balancing)
    Result streamk_dp       = run<DeviceGemmStreamK<8,4>>("StreamK emulating basic data-parallel GEMM", options);
    options.avail_sms       = -1;       // Reset loadbalancing width to unspecified SMs (i.e., the number of device SMs)

    printf("  Speedup vs Basic-DP: %.3f\n", (basic_dp.avg_runtime_ms / streamk_dp.avg_runtime_ms));

    options.split_k_factor++;     // Increment splitting factor for next evaluation

    */
  }
  /*
  // Show that StreamK can emulate "Split-K" with a tile-splitting factor
  Result basic_splitk = run<DeviceGemmBasic>(
    std::string("Basic split-K GEMM with tile-splitting factor ") + std::to_string(options.split_k_factor),
    options);

  Result streamk_splitk = run<DeviceGemmStreamK<8,4>>(
    std::string("StreamK emulating Split-K GEMM with tile-splitting factor ") + std::to_string(options.split_k_factor),
    options);

  printf("  Speedup vs Basic-SplitK: %.3f\n", (basic_splitk.avg_runtime_ms / streamk_splitk.avg_runtime_ms));
  */
  return 0;
}
