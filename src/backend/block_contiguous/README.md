## Kernels
1. Naive: {blocked/non_blocked}_gemm.cu
2. Vectorised loads: {blocked/non_blocked}_gemm_vec2.cu
3. 1D Tiling: {blocked/non_blocked}_gemm_vec_1d2.cu
4. 2D Tiling: {blocked/non_blocked}_gemm_vec_2d2.cu
5. Warp tiling: {blocked/non_blocked}_gemm_vec_warp2.cu
6. Double Buffering: blocked_gemm_vec_double_buf2.cu, non_blocked_gemm_vec_double_buf2_async.cu, non_blocked_gemm_vec_double_buf2_sync.cu
7. CUTLASS baseline: basic_cutlass.cu

## Compilation

### Debug Mode
Verifies the correctness of the kernel. Unset by default.
```
export CUTLASS="/home/shgoel/dphpc/cutlass/cutlass/include"
nvcc -DCPU_DEBUG=1 -o test <kernel> -lcublas -I${CUTLASS} --std=c++17 -arch sm_120
```

### Benchmarking Mode
Runs 3 warmup iterations followed by 50 benchmarking iterations to calculate average TFLOP/s and 95% confidence intervals. Set by default.

```
export CUTLASS="/home/shgoel/dphpc/cutlass/cutlass/include"
nvcc -DBENCHMARK=1 -o test <kernel> -lcublas -I${CUTLASS} --std=c++17 -arch sm_120
```

## Testing
```
./test <M> <N> <K>
```
