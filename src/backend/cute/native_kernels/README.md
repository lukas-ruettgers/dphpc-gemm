# CUDA Matrix Multiplication Testing

## Compilation

Compile `testing.cu` with:

```bash
nvcc -o test testing.cu -lcublas -I/home/apaladi/dphpc/cutlass/include --std=c++17 -arch sm_120
```

## Running Tests

### Basic execution:

```bash
./test <kernel> <M> <N> <K> <BM> <BN> <BK> [--check]
```

**Available kernels:** cpasync, cpasync_bk32, vector, scalar, pipelined

## Parameters
* `kernel` - Kernel name from the list above
* `M`, `N`, `K` - Matrix dimensions (C = A × B, where A is M×K, B is K×N, C is M×N)
* `BM`, `BN`, `BK` - Block/tile dimensions
* `--check` - Optional flag to verify correctness against reference implementation

**Note:** The cpasync kernel with BK=32 should use cpasync_bk32 as the kernel name.
                     

### Example:

```bash
./test cpasync_bk32 1024 1024 1024 128 128 32 --check
```

## Profiling with NCU

### Full profile:

```bash
ncu --set full ./test <kernel> <M> <N> <K> <BM> <BN> <BK> [--check]
```

### Skip warm-up launches:

```bash
ncu --set full --launch-skip 20 ./test <kernel> <M> <N> <K> <BM> <BN> <BK>
```

### Limit number of launch profiles collected:

```bash
ncu --set full --launch-skip 20 --launch-count <N> ./test <kernel> <M> <N> <K> <BM> <BN> <BK>
```


