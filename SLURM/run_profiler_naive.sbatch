#!/bin/bash

#SBATCH --time=02:59
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --output=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/naive_prof%j.log
#SBATCH --error=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/naive_prof%j.err

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
     /home/shgoel/dphpc/cutlass/benchmark_gemm/src/non_blocked_gemm_vec.cu \
     -o ./non_blocked_gemm

ncu \
  --set=@roofline \
  --section=SpeedOfLight \
  --section=MemoryWorkloadAnalysis_Tables \
  --metrics=l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \
  --export=targeted_analysis_naive_gemm \
  ./non_blocked_gemm 4096 4096 4096 16 16 16 

ncu --import ./targeted_analysis_naive_gemm.ncu-rep --page details > ../prof_output/details_naive_gemm.txt



