# Baseline
`gemm_universal` provides the CUTLASS baseline optimized by NVIDIA. We only have to specify the problem parameters, no performance parameters.

`gemm_config` allows us to explicitly configure performance parameters such as block shapes etc.

# Tile size profiling

### COMPILE
Make sure to `cd` to the root directory `dphpc-gemm`.
#### Configure (once) with your desired defaults:
```
cmake -S /home/$USER/dphpc/dphpc-gemm/src/backend/cute -B /home/$USER/dphpc/dphpc-gemm/build \
  -DCUTLASS_ROOT=/home/$USER/dphpc/dphpc-gemm/external/cutlass \
  -DDATATYPE_A=half -DDATATYPE_B=half -DDATATYPE_C=float -DDATATYPE_ACC=float \
  -DLAYOUT_A=row -DLAYOUT_B=col -DLAYOUT_C=row \
  -DTB_M=128 -DTB_N=128 -DTB_K=32 -DWP_M=64 -DWP_N=64 -DWP_K=32 \
  -DINST_M=16 -DINST_N=8 -DINST_K=16 -DSTAGES=3 \
  -DALIGN_A=8 -DALIGN_B=8 -DSPLITK_SERIAL=1 -DARCH_SM=120 \
  -DUSE_TENSOR_OP=1 -DUSE_TF32=0
```

##### Compile-time flags 
E.g. to run FP32 SIMT, change the -D flags to
```
    ...
  -DDATATYPE_A=float -DDATATYPE_B=float -DDATATYPE_C=float -DDATATYPE_ACC=float \
  -DLAYOUT_A=row -DLAYOUT_B=col -DLAYOUT_C=row \
  -DTB_M=128 -DTB_N=128 -DTB_K=32 -DWP_M=64 -DWP_N=64 -DWP_K=32 \
  -DINST_M=1 -DINST_N=1 -DINST_K=1 -DSTAGES=3 -DUSE_TENSOR_OP=0 \
  -DALIGN_A=4 -DALIGN_B=4 -DSPLITK_SERIAL=0 -DARCH_SM=120 \
    ...
```

#### Actual compilation
```
cmake --build /home/$USER/dphpc/dphpc-gemm/build -j
```
#### Run (example)
```
./build/gemm_baseline 4096 4096 4096 1 1.0 0.0
```

#### Compile with nvcc directly
1. FP16 Tensor Cores (typical baseline):
```
nvcc -O3 -std=c++17 -lineinfo \
     -gencode arch=compute_120,code=sm_120 \
     -gencode arch=compute_120,code=compute_120 \
     -I"/home/$USER/dphpc/dphpc-gemm/external/cutlass/include" \
     -I"/home/$USER/dphpc/dphpc-gemm/external/cutlass/tools/util/include/" \
     -I"$CUDA_HOME/include" \
     -L"$CUDA_HOME/lib64" \
     -Xptxas=-v \
     -lnvToolsExt \
     -DCUTLASS_ENABLE_TENSOR_OP_MATH=1 \
     -DDTYPE_A=1 -DDTYPE_B=1 -DDTYPE_C=2 -DDTYPE_ACC=2 \
     -DLAYOUT_A_CODE=0 -DLAYOUT_B_CODE=1 -DLAYOUT_C_CODE=0 \
     -DTB_M=128 -DTB_N=128 -DTB_K=32 -DWP_M=64 -DWP_N=64 -DWP_K=32 \
     -DINST_M=16 -DINST_N=8 -DINST_K=16 -DSTAGES=3 -DUSE_TENSOR_OP=1 \
     -DALIGN_A=8 -DALIGN_B=8 \
     -DSPLITK_SERIAL=0 \
     -DARCH_SM=120 \
     /home/$USER/dphpc/dphpc-gemm/src/backend/cute/gemm_baseline.cu \
     -o /home/$USER/dphpc/dphpc-gemm/build/gemm_baseline.out
```

###### Debug
```
nvcc -O3 -std=c++17 -lineinfo \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120 \
  -I"/home/$USER/dphpc/dphpc-gemm/external/cutlass/include" \
  -I"/home/$USER/dphpc/dphpc-gemm/external/cutlass/tools/util/include/" \
  -I"$CUDA_HOME/include" \
  -L"$CUDA_HOME/lib64" \
  -Xptxas=-v \
  -lnvToolsExt \
  /home/$USER/dphpc/dphpc-gemm/src/backend/cute/gemm_blackwell.cu \
  -o /home/$USER/dphpc/dphpc-gemm/build/gemm_universal.out
```