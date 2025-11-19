#!/bin/bash

#SBATCH --time=00:59
#SBATCH --account=dphpc
#SBATCH --gpus=5060ti:1
#SBATCH --output=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/%j.log
#SBATCH --error=/home/shgoel/dphpc/cutlass/benchmark_gemm/output/%j.err

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
cmake ..
make -j

./cutlass_gemm_tunable --M 8192 --N 8192 --K 64 --autotune
                            