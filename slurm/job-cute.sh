#!/bin/bash
#SBATCH --job-name=dphpc-gemm-cute
#SBATCH --mail-type=NONE # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/work/scratch/$USER/dphpc/jobs/%x_%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/work/scratch/$USER/dphpc/jobs/%x_%j.err # where to store error messages
#SBATCH --nodes=1
#SBATCH --gres=gpu:5060ti:1 # Compute Capability of 1080 Ti is too low.
#SBATCH --time=0-01:00 # formats: days-hours:minutes, days-hours, hours:minutes, minutes

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

cd /home/$USER/dphpc/dphpc-gemm
./build/a.out
