#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account=dphpc
#SBATCH --output=nsys.log

# nsys profile \
#   --gpu-metrics-devices=all \
#   --trace=cuda,osrt,nvtx \
#   --trace-fork-before-exec=true \
#   --cuda-memory-usage=true \
#   --cuda-um-cpu-page-faults=true \
#   --cuda-um-gpu-page-faults=true \
#   --opengl-gpu-workload=false \
#   --delay=2 \
#   --duration=30 \
#   --sample=cpu \
#   --cpuctxsw=process-tree \
#   --output=nsys.log \
#   --force-overwrite=true \
#   ./cutlass_gemm_tunable2

nsys profile \
  --trace=cuda,nvtx \
  --output=nsys.log \
  --stats true \
  --force-overwrite=true \
  ./cutlass_gemm_tunable2
