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

M=(1024 2048 4096 8192)
N=(1024 2048 4096 8192)
K=(32 64 128 256)

ThreadBlock=("128x256x32" "256x128x32" "128x128x32")
WarpBlock=("64x64x32")
InstBlock=("8x8x4" "16x8x8")

for m in "${M[@]}"; do
    for n in "${N[@]}"; do
        for k in "${K[@]}"; do
            for TB in "${ThreadBlock[@]}"; do
                for WB in "${WarpBlock[@]}"; do
                    for inst in "${InstBlock[@]}"; do
                        ./cutlass_gemm_tunable \
                            --M "$m" \
                            --N "$n" \
                            --K "$k" \
                            --threadblock "$TB" \
                            --warp "$WB" \
                            --inst "$inst"
                    done
                done
            done
        done
    done
done