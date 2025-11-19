```
module purge
module load cuda/13.0
module save default

# Clean env
export CUDA_HOME=/cluster/data/cuda/13.0.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LD_PRELOAD
export CUDACXX=${CUDA_HOME}/bin/nvcc
```


## Build

- Modify the path to CUTLASS repo in `CMakeLists.txt` (line 10)

```
mkdir -p build && cd build
cmake ..
make -j

./cutlass_gemm
./wmma_gemm
```

## Autotuner
```
./cutlass_gemm_tunable --M 8192 --N 8192 --K 64 --autotune --iters 100
```
(see also SLURM/run_autotune.sbatch)