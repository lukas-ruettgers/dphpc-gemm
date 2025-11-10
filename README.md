# PO2 GEMM Test

This folder contains a simple GEMM test using CUTLASS.

## Build Instructions

```bash
cd po2
# Compile with your own CUTLASS path
nvcc -O3 -std=c++17 testing.cu -o testnew -I/your/path/cutlass/include -arch=sm_120 -lcublas
# By default, the test runs with a matrix size of 256³, but you can specify your own dimensions
./test 1024 1024 1024

