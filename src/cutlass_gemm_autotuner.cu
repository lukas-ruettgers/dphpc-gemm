#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/gemm.h"

using ElementAccumulator = float;
using ElementCompute = float;
using ElementOutput = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

static void checkCuda(cudaError_t res, const char* msg = "") {
  if (res != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(res) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////

// Helper to allocate and initialize host/device memory
template <typename T>
T *alloc_device(size_t count) {
  T *dptr = nullptr;
  checkCuda(cudaMalloc((void**)&dptr, count * sizeof(T)), "cudaMalloc");
  return dptr;
}

template <typename T>
T *alloc_host(size_t count) {
  T *h = (T*)malloc(sizeof(T)*count);
  if (!h) { std::cerr<<"host malloc failed\n"; std::exit(EXIT_FAILURE); }
  return h;
}

// Fill with simple pattern
template <typename T>
void fill_random(T* ptr, size_t n) {
  for (size_t i=0;i<n;i++) {
    float v = (float)( (rand() % 100) - 50 ) / 23.0f;
    ptr[i] = (T)v;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////



// Enum for all available configurations with descriptive names
enum class CutlassConfig
{
    TB_128x256x64_W_64x64x64_S3 = 0,
    TB_64x256x32_W_32x64x32_S4 = 1,
    TB_128x128x32_W_64x64x32_S4 = 2,
    TB_128x64x32_W_64x32x32_S4 = 3,
    TB_64x128x32_W_32x64x32_S4 = 4,
    TB_128x32x32_W_64x32x32_S4 = 5,
    TB_64x32x32_W_32x32x32_S5 = 6,
    TB_32x64x32_W_32x32x32_S5 = 7,
    TB_128x128x64_W_64x64x64_S4 = 8,
    TB_128x64x64_W_64x32x64_S4 = 9,
    TB_64x128x64_W_32x64x64_S4 = 10,
    TB_256x256x32_W_64x64x32_S3 = 11,
    TB_256x128x32_W_64x64x32_S3 = 12,
    TB_128x256x32_W_64x64x32_S3 = 13,
    TB_64x64x32_W_32x32x32_S5 = 14,
    TB_256x256x64_W_64x64x64_S3 = 15,
    TB_256x128x64_W_64x64x64_S3 = 16,
    TB_128x256x64_W_64x64x64_S4 = 17,
    TB_256x256x64_W_64x64x64_S4 = 18,
    TB_128x128x64_W_64x64x64_S3 = 19,
    Count // to get the number of configurations
};

template <int ThreadBlockM, int ThreadBlockN, int ThreadBlockK,
          int WarpM, int WarpN, int WarpK,
          int InstrM, int InstrN, int InstrK,
          int Stages, typename InputElementType>
struct CutlassGemmAutotuneConfig
{
    using ElementInput = InputElementType;

    using ThreadBlockShape = cutlass::gemm::GemmShape<ThreadBlockM, ThreadBlockN, ThreadBlockK>;
    using WarpShape = cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>;
    using InstructionShape = cutlass::gemm::GemmShape<InstrM, InstrN, InstrK>;

    static constexpr int kStages = Stages;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value>;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput,
        LayoutA,
        ElementInput,
        LayoutB,
        ElementOutput,
        LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, // SM80 (compatible with Ampere/Ada Lovelace)
        ThreadBlockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        kStages>;
};

template <typename Config>
cudaError_t cutlass_gemm_autotune_launch(
    int M, int N, int K,
    const typename Config::ElementInput *d_A, int lda,
    const typename Config::ElementInput *d_B, int ldb,
    ElementOutput *d_C, int ldc,
    float alpha, float beta,
    cudaStream_t stream = nullptr)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    typename Config::Gemm gemm_op;

    typename Config::Gemm::Arguments args(
        {M, N, K},
        {d_A, lda},
        {d_B, ldb},
        {d_C, ldc},
        {d_C, ldc},
        {alpha, beta});

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorNotSupported;

    status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;

    return cudaSuccess;
}

template <int BM, int BN, int BK,
          int WM, int WN, int WK,
          int IM, int IN, int IK,
          int STAGES, typename T>
using GemmCfg = CutlassGemmAutotuneConfig<BM, BN, BK, WM, WN, WK, IM, IN, IK, STAGES, T>;

struct GemmConfigEntry
{
    int BM, BN, BK;
    int WM, WN, WK;
    int IM, IN, IK;
    int stages;
};

constexpr GemmConfigEntry kConfigs[] = {
    {128, 256, 64, 64, 64, 64, 16, 8, 16, 3},
    {64, 256, 32, 32, 64, 32, 16, 8, 16, 4},
    {128, 128, 32, 64, 64, 32, 16, 8, 16, 4},
    {128, 64, 32, 64, 32, 32, 16, 8, 16, 4},
    {64, 128, 32, 32, 64, 32, 16, 8, 16, 4},
    {128, 32, 32, 64, 32, 32, 16, 8, 16, 4},
    {64, 32, 32, 32, 32, 32, 16, 8, 16, 5},
    {32, 64, 32, 32, 32, 32, 16, 8, 16, 5},
    {128, 128, 64, 64, 64, 64, 16, 8, 16, 4},
    {128, 64, 64, 64, 32, 64, 16, 8, 16, 4},
    {64, 128, 64, 32, 64, 64, 16, 8, 16, 4},
    {256, 256, 32, 64, 64, 32, 16, 8, 16, 3},
    {256, 128, 32, 64, 64, 32, 16, 8, 16, 3},
    {128, 256, 32, 64, 64, 32, 16, 8, 16, 3},
    {64, 64, 32, 32, 32, 32, 16, 8, 16, 5},
    {256, 256, 64, 64, 64, 64, 16, 8, 16, 3},
    {256, 128, 64, 64, 64, 64, 16, 8, 16, 3},
    {128, 256, 64, 64, 64, 64, 16, 8, 16, 4},
    {256, 256, 64, 64, 64, 64, 16, 8, 16, 4},
    {128, 128, 64, 64, 64, 64, 16, 8, 16, 3},
};

template <int IDX, typename T>
struct GetConfig
{
    static constexpr auto cfg = kConfigs[IDX];
    using type = GemmCfg<
        cfg.BM, cfg.BN, cfg.BK,
        cfg.WM, cfg.WN, cfg.WK,
        cfg.IM, cfg.IN, cfg.IK,
        cfg.stages, T>;
};

template <typename CutlassType, int IDX>
cudaError_t dispatch_config(
    int M, int N, int K,
    const CutlassType *d_A, int lda,
    const CutlassType *d_B, int ldb,
    ElementOutput *d_C, int ldc,
    float alpha, float beta,
    cudaStream_t stream)
{
    using FP16Cfg = typename GetConfig<IDX, cutlass::half_t>::type;
    using BF16Cfg = typename GetConfig<IDX, cutlass::bfloat16_t>::type;

    if constexpr (std::is_same_v<CutlassType, cutlass::half_t>)
        return cutlass_gemm_autotune_launch<FP16Cfg>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    else
        return cutlass_gemm_autotune_launch<BF16Cfg>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
}

template <typename TorchType, typename CutlassType>
cudaError_t dispatch_cutlass_autotune(
    CutlassConfig config,
    const int M, const int N, const int K,
    const CutlassType *d_A, int lda,
    const CutlassType *d_B, int ldb,
    ElementOutput *d_C, int ldc,
    const float alpha, const float beta,
    cudaStream_t stream = nullptr)
{
    auto launch = [&](auto I)
    {
        return dispatch_config<CutlassType, I>(M, N, K, d_A, lda, d_B, ldb, d_C, ldc, alpha, beta, stream);
    };

    switch (static_cast<int>(config))
    {
#define CASE_CONFIG(I) \
    case I:            \
        return launch(std::integral_constant<int, I>{});
        CASE_CONFIG(0)
        CASE_CONFIG(1)
        CASE_CONFIG(2)
        CASE_CONFIG(3)
        CASE_CONFIG(4)
        CASE_CONFIG(5)
        CASE_CONFIG(6)
        CASE_CONFIG(7)
        CASE_CONFIG(8)
        CASE_CONFIG(9)
        CASE_CONFIG(10)
        CASE_CONFIG(11)
        CASE_CONFIG(12)
        CASE_CONFIG(13)
        CASE_CONFIG(14)
        CASE_CONFIG(15)
        CASE_CONFIG(16)
        CASE_CONFIG(17)
        CASE_CONFIG(18)
        CASE_CONFIG(19)
    default:
        return cudaErrorInvalidValue;
#undef CASE_CONFIG
    }
}


// Function to get the number of available configs
int get_num_cutlass_configs()
{
    return static_cast<int>(CutlassConfig::Count);
}

