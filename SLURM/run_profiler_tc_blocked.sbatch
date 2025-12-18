#!/bin/bash

#SBATCH --time=00:59
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --output=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/tc_blocked_prof%j.log
#SBATCH --error=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/tc_blocked_prof%j.err

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

mkdir -p build_prof && cd build_prof
rm -rf *
nvcc -O3 -lineinfo \
     -gencode arch=compute_120,code=sm_120 \
     -gencode arch=compute_120,code=compute_120 \
     -DCUTLASS_ENABLE_TENSOR_OP_MATH=1 \
     -I"/home/shgoel/dphpc/cutlass/cutlass/include" \
     -I"/home/shgoel/dphpc/cutlass/cuda-samples/Common" \
     -I"/home/shgoel/dphpc/cutlass/benchmark_gemm/include" \
     -I"$CUDA_HOME/include" \
     -L"$CUDA_HOME/lib64" -lnvToolsExt \
     /home/shgoel/dphpc/cutlass/benchmark_gemm/src/blocked_gemm_tc.cu \
     -o ./blocked_gemm_tc

ncu \
  --set=@roofline \
  --section=InstructionStats \
  --section=LaunchStats \
  --section=Occupancy \
  --section=SpeedOfLight \
  --section=WarpStateStats \
  --metrics=sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st \
  --export=targeted_analysis_blocked_gemm_tc \
  ./blocked_gemm_tc

ncu --import ./targeted_analysis_blocked_gemm_tc.ncu-rep --page details > ../prof_output/details_blocked_gemm_tc.txt



