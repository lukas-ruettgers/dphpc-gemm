# COMPILE       |                           |


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

### Compile flags for profiling
```
nvcc -O3 -lineinfo \
     -gencode arch=compute_120,code=sm_120 \
     -gencode arch=compute_120,code=compute_120 \
     -DCUTLASS_ENABLE_TENSOR_OP_MATH=1 \
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
| `-gencode arch=compute_XXX,code=compute_XXX` | Embed PTX for forward compatibility                | Portability    

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
[Ref for profiling basics](https://puzzles.modular.com/puzzle_30/nvidia_profiling_basics.html)

## Pitfalls
**Critical**: For accurate profiling, build with full debug information
// TODO

## Best Practices
[Ref](https://puzzles.modular.com/puzzle_30/nvidia_profiling_basics.html#profiling-workflow-best-practices)
1. Progress from `--output=quick_look` to `--output=detailed`
2. Multi-run analysis with `diff`
3. Build with full debug information for the profiler

| Flag        | Purpose                                                                                 | Effect on performance                                                |
| ----------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `-G`        | Generate *device-side* debug information (symbols, variables, full line mappings).      | **Disables most optimizations** — slow but gives full symbolic info. |
| `-g`        | Generate *host-side* debug symbols for CPU code (passed to host compiler).              | Negligible impact.                                                   |
| `-lineinfo` | Embed source line correlation info in the device code (for profilers only, no symbols). | **Keeps optimizations**, ideal for release-mode profiling.           |


## NSight Systems
More holistic:
- host-device interaction (memory transfer) 
- CUDA API calls

### 1. Run the profiler
```
nsys profile --output myrun \
             --stats true \
             --trace cuda,nvtx \
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


#### Advanced NSight Systems profiling
For comprehensive system-wide analysis, use these advanced nsys flags:

```
nsys profile \
  --gpu-metrics-devices=all \
  --trace=cuda,osrt,nvtx \
  --trace-fork-before-exec=true \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  --opengl-gpu-workload=false \
  --delay=2 \
  --duration=30 \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --output=comprehensive_profile \
  --force-overwrite=true \
  ./build/a.out
```

##### Flag explanations:

--gpu-metrics-devices=all: Collect GPU metrics from all devices
--trace=cuda,osrt,nvtx: Comprehensive API tracing
--cuda-memory-usage=true: Track memory allocation/deallocation
--cuda-um-cpu/gpu-page-faults=true: Monitor Unified Memory page faults
--delay=2: Wait 2 seconds before profiling (avoid cold start)
--duration=30: Profile for 30 seconds max
--sample=cpu: Include CPU sampling for hotspot analysis
--cpuctxsw=process-tree: Track CPU context switches

### 2. Inspect results
```
nsys stats --report summary        myrun.nsys-rep > MYSTATSFILE.txt
nsys stats --report gpukernsum     myrun.nsys-rep > MYSTATSFILE.txt  # kernel time table
nsys stats --report cudaapisum     myrun.nsys-rep > MYSTATSFILE.txt  # CUDA API call summary
nsys stats --report nvtxsum        myrun.nsys-rep > MYSTATSFILE.txt  # NVTX ranges (if you used NVTX markers)
```

## NSight Compute
GPU kernel in-depth profiling
- Roofline model analysis
- Memory hierarchy utilization: smem, registers
- Warp execution efficiency
- Compute unit utilization

#### Advanced NSight Compute profiling 
For detailed kernel analysis with comprehensive metrics: [(REF)](https://puzzles.modular.com/puzzle_30/nvidia_profiling_basics.html#advanced-nsight-compute-profiling)


### PROFILE BANK CONFLICTS
```
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st ./build/a.out

# Additional context metrics
ncu --metrics=smsp__sass_average_branch_targets_threads_uniform.pct ./build/a.out
ncu --metrics=smsp__warps_issue_stalled_membar_per_warp_active.pct ./build/a.out
```

Ref: [here](https://puzzles.modular.com/puzzle_32/shared_memory_bank.html).

## Check register spills
1. Run `nvcc` with `-Xptxas=-v`.
2. In the output, there'll be lines that look like:
```
ptxas info    : Compiling entry function '_ZN7cutlass4gemm...' for 'sm_80'
ptxas info    : Function properties for _ZN7cutlass4gemm...
    128 registers
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

### Force register spillage (for experimental insights only)
Add `-maxrregcount=64` to `nvcc`

## Profiling with Visual Studio
1. Set up the SSH connection [(Ref)](https://learn.microsoft.com/en-us/cpp/linux/connect-to-your-remote-linux-computer?view=msvc-170#set-up-the-remote-connection)
2. Create a CMake Linux project in Visual Studio [(Ref)](https://learn.microsoft.com/en-us/cpp/linux/cmake-linux-project?view=msvc-170)

# BUILD
```
mkdir -p build && cd build
cmake ..
cmake --build . -j
./gemm  # your args
```
