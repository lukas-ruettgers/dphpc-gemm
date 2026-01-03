#!/bin/bash

#SBATCH --time=00:59
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --output=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/%j_blocked.log
#SBATCH --error=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/%j_blocked.err

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
     /home/shgoel/dphpc/cutlass/benchmark_gemm/src/blocked_gemm_vec_warp.cu \
     -o ./blocked_gemm.cu

./blocked_gemm.cu

