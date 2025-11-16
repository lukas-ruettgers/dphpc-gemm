nvcc -O3 -lineinfo \
     -gencode arch=compute_120,code=sm_120 \
     -gencode arch=compute_120,code=compute_120 \
     -DCUTLASS_ENABLE_TENSOR_OP_MATH=1 \
     -I"/home/$USER/dphpc/cutlass/include" \
     -I"/home/$USER/dphpc/cutlass/tools/util/include/" \
     -I"$CUDA_HOME/include" \
     -I"/home/$USER/dphpc/dphpc-gemm/include" \
     -L"$CUDA_HOME/lib64" -lnvToolsExt \
     /home/$USER/dphpc/dphpc-gemm/src/cutlass_gemm_tunable2.cu \
     -o /home/$USER/dphpc/dphpc-gemm/cutlass_gemm_tunable2