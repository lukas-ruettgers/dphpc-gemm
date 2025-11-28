#!/bin/bash

#SBATCH --time=00:59
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --output=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/%j_profile.log
#SBATCH --error=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/%j_profile.err

. /etc/profile.d/modules.sh
module purge
module load cuda/13.0
module save default

# Clean env
export CUDA_HOME=/cluster/data/cuda/13.0.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LD_PRELOAD
export CUDACXX=${CUDA_HOME}/bin/nvcc

mkdir -p build && cd build
rm -rf *
nvcc -O3 -lineinfo \
     -gencode arch=compute_120,code=sm_120 \
     -gencode arch=compute_120,code=compute_120 \
     -DCUTLASS_ENABLE_TENSOR_OP_MATH=1 \
     -I"/home/shgoel/dphpc/cutlass/cutlass/include" \
     -I"/home/shgoel/dphpc/cutlass/cutlass/tools/util/include" \
     -I"/home/shgoel/dphpc/cutlass/benchmark_gemm/include" \
     -I"$CUDA_HOME/include" \
     -L"$CUDA_HOME/lib64" -lnvToolsExt \
     /home/shgoel/dphpc/cutlass/benchmark_gemm/src/cutlass_gemm_tunable2.cu \
     -o ./cutlass_gemm_tunable


ncu \
  --set=@roofline \
  --section=InstructionStats \
  --section=LaunchStats \
  --section=Occupancy \
  --section=SpeedOfLight \
  --section=WarpStateStats \
  --metrics=sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st \
  --export=targeted_analysis_32 \
  ./cutlass_gemm_tunable --M 8192 --N 8192 --K 32  --threadblock "64x128x64" --warp "32x64x64" --inst "16x8x16" --stages 4

  ncu --import ./targeted_analysis_32.ncu-rep --page details > ../prof_output/details_32_CUTLASS.txt

ncu \
  --set=@roofline \
  --section=InstructionStats \
  --section=LaunchStats \
  --section=Occupancy \
  --section=SpeedOfLight \
  --section=WarpStateStats \
  --metrics=sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st \
  --export=targeted_analysis_64 \
  ./cutlass_gemm_tunable --M 8192 --N 8192 --K 64  --threadblock "64x256x32" --warp "32x64x32" --inst "16x8x16" --stages 4

  ncu --import ./targeted_analysis_64.ncu-rep --page details > ../prof_output/details_64_CUTLASS.txt

ncu \
  --set=@roofline \
  --section=InstructionStats \
  --section=LaunchStats \
  --section=Occupancy \
  --section=SpeedOfLight \
  --section=WarpStateStats \
  --metrics=sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st \
  --export=targeted_analysis_128 \
  ./cutlass_gemm_tunable --M 8192 --N 8192 --K 128  --threadblock "64x256x32" --warp "32x64x32" --inst "16x8x16" --stages 4

  ncu --import ./targeted_analysis_128.ncu-rep --page details > ../prof_output/details_128_CUTLASS.txt

ncu \
  --set=@roofline \
  --section=InstructionStats \
  --section=LaunchStats \
  --section=Occupancy \
  --section=SpeedOfLight \
  --section=WarpStateStats \
  --metrics=sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st \
  --export=targeted_analysis_256 \
  ./cutlass_gemm_tunable --M 8192 --N 8192 --K 256  --threadblock "128x64x32" --warp "64x32x32" --inst "16x8x16" --stages 4

  ncu --import ./targeted_analysis_256.ncu-rep --page details > ../prof_output/details_256_CUTLASS.txt

ncu \
  --set=@roofline \
  --section=InstructionStats \
  --section=LaunchStats \
  --section=Occupancy \
  --section=SpeedOfLight \
  --section=WarpStateStats \
  --metrics=sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st \
  --export=targeted_analysis_512 \
  ./cutlass_gemm_tunable --M 8192 --N 8192 --K 512  --threadblock "128x64x32" --warp "64x32x32" --inst "16x8x16" --stages 4

  ncu --import ./targeted_analysis_512.ncu-rep --page details > ../prof_output/details_512_CUTLASS.txt


                            