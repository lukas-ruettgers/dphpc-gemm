#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account=dphpc
#SBATCH --output=sbatch_ncu_benchmark_output.log

# Generate ncu data
# ncu --set detailed -o ncu_analysis --force-overwrite <binary>

# Get info
# ncu --import ncu_analysis.ncu-rep --print-details all > ncu_analysis

# CSV: --csv
# Roofline, throughput: --section SpeedOfLight_RooflineChart
# Runtime: --section SpeedOfLight
# Skip warmup profiling: --launch-skip <number of warmup runs>

# Arg 1: executable

ncu \
  --set detailed \
  -o ncu_benchmark_output \
  --force-overwrite \
  $@