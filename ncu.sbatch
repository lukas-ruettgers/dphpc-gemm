#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account=dphpc
#SBATCH --output=ncu.log

# Visual friendly
# ncu --import ncu_analysis.ncu-rep --print-details all --section SpeedOfLight_RooflineChart > ncu_analysis

# CSV parsing friendly
# ncu --import ncu_analysis.ncu-rep --csv --print-details all --section SpeedOfLight_RooflineChart > ncu_analysis

ncu \
  --set full \
  -o ncu_analysis \
  --force-overwrite \
  ./cutlass_gemm_tunable2
