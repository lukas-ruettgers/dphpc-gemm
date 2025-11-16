#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account=dphpc
#SBATCH --output=ncu.log

# Generate ncu data
# ncu --set full -o ncu_analysis --force-overwrite <binary>

# Get info
# ncu --import ncu_analysis.ncu-rep --print-details all > ncu_analysis

# CSV: --csv
# Roofline, throughput: --section SpeedOfLight_RooflineChart
# Runtime: --section SpeedOfLight
# Skip warmup profiling: --launch-skip <number of warmup runs>

ncu \
  --set detailed \
  -o ncu_analysis \
  --force-overwrite \
  ./cutlass_gemm_tunable2
