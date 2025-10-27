# COMPILE
### COMPILE FOR 5060 explicitly
```
nvcc -g -G -lineinfo \
     -gencode arch=compute_120,code=sm_120 \
     -gencode arch=compute_120,code=compute_120 \
     -I"/home/$USER/dphpc/dphpc-gemm/external/cutlass/include" \
     -I"/home/$USER/dphpc/dphpc-gemm/external/cutlass/tools/util/include/" \
     -I"$CUDA_HOME/include" \
     -L"$CUDA_HOME/lib64" -lnvToolsExt \
     /home/$USER/dphpc/dphpc-gemm/src/backend/cute/transpose.cu \
     -o /home/$USER/dphpc/dphpc-gemm/build/a.out
```

| Option                                       | Purpose                                            | Recommended for       |                           |
| :------------------------------------------- | :------------------------------------------------- | :-------------------- | ------------------------- |
| `grep -E`                                    | Use extended regex syntax (no need to escape `     | `)                    | Flexible pattern matching |
| `-g`                                         | Debug info for host (CPU)                          | Debugging / profiling |                           |
| `-G`                                         | Debug info for device (GPU), disables optimization | GPU debugging         |                           |
| `-lineinfo`                                  | Keeps line info without disabling optimizations    | Profiling             |                           |
| `-gencode arch=compute_XXX,code=sm_XXX`      | Generate native GPU binary                         | Running efficiently   |                           |
| `-gencode arch=compute_XXX,code=compute_XXX` | Embed PTX for forward compatibility                | Portability           |                           |


## COMPILE FOR 5060 Ti (CC 12.0)
Switch to a newer CUDA version
```
module purge
module load cuda/13.0
module save default

# Clean env
export CUDA_HOME=/cluster/data/cuda/13.0.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LD_PRELOAD
```

# SLURM
## LIST NODES WITH FEATURES
```
sinfo -N -o "%15N %10G %20f %8P %10t"
```
## SRUN
There are `studgpu-node[01-32]`.

### 5060
Set one of these nodes:
```
srun -A dphpc --nodelist=studgpu-node01 --pty bash -i 
srun -A dphpc --nodelist=studgpu-node09 --pty bash -i 
srun -A dphpc --nodelist=studgpu-node17 --pty bash -i 
srun -A dphpc --nodelist=studgpu-node25 --pty bash -i 
```

### 1080
All other nodes are 1080s.

## SBATCH
The `module` command is not available for running jobs unless you load the following to your batch script:
```
. /etc/profile.d/modules.sh
module add cuda/12.9
```
[Ref](https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterCuda)

# Profiling
## NSight Systems
More holistic, restricted to host
```
nsys profile --output myrun \
             --stats true \
             --trace cuda,nvtx,osrt,mpi \
             ./a.out
```
| Option                   | Purpose                                                                      |
| :----------------------- | :--------------------------------------------------------------------------- |
| `--output <name>`        | base filename for report (`.nsys-rep` appended automatically)                |
| `--trace <domains>`      | choose trace domains: `cuda`, `nvtx`, `osrt`, `mpi`, `openmp`, `cudnn`, etc. |
| `--stats true`           | generate a `.nsys-stats` file automatically                                  |
| `--sample=none`          | disable CPU sampling (useful if cluster disables it)                         |
| `--duration <sec>`       | limit profiling duration                                                     |
| `--force-overwrite=true` | overwrite existing reports                                                   |

### Inspect results
```
nsys stats --report summary        myrun.nsys-rep
nsys stats --report gpukernsum     myrun.nsys-rep   # kernel time table
nsys stats --report cudaapisum     myrun.nsys-rep   # CUDA API call summary
nsys stats --report nvtxsum        myrun.nsys-rep   # NVTX ranges (if you used NVTX markers)
```

## NSight Compute
GPU kernel in-depth profiling

### PROFILE BANK CONFLICTS
```
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st ./build/a.out

# Additional context metrics
ncu --metrics=smsp__sass_average_branch_targets_threads_uniform.pct ./build/a.out
ncu --metrics=smsp__warps_issue_stalled_membar_per_warp_active.pct ./build/a.out
```

Ref: [here](https://puzzles.modular.com/puzzle_32/shared_memory_bank.html).

# BUILD
```
mkdir -p build && cd build
cmake ..
cmake --build . -j
./gemm  # your args
```
