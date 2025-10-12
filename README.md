# Repository structure
```
dphpc-gemm/
├─ config/                              # User-configurable environment
│  ├─ config.example.sh
│  └─ (config.sh)                       # user-local, git-ignored
│
├─ external/                            # External dependencies (submodules ok)
│  └─ cutlass/
│
├─ slurm/                               # Slurm job files
│  └─ job-cute.sh
│
├─ src/
│  ├─ core/
│  │  ├─ problem.hpp/cpp                # main(): parse CLI, fetch hardware specs, invoke planner
│  │  ├─ plan.hpp                       # Backend-agnostic hyperparameters (e.g. backend, dtypes, tiles)
│  │  ├─ planner.hpp                    # Search optimal hyperparameter combination for particular problem (header-only for now)
│  │  ├─ dispatcher.cpp/hpp             # Dispatch to a backend with specific plan                
│  │  ├─ device_query.cpp/hpp           # Fetch GPU hardware specificiation
│  │
│  ├─ backend/
│  │  └─ cute/
│  │     ├─ plan.hpp                    # Backend-specific hyperparameters (path NT/TN, blk_m/n/k, layouts, overrides)
│  │     ├─ backend_adaptor.cpp/hpp     # Interface for dispatcher to kernel launch
│  │     └─ gemm.cu                     # CUDA kernel and kernel launch
│  │
│  ├─ utils/
│  │  ├─ cuda_check.hpp                 # CUDA_CHECK, CUDA_CHECK_LAST, CUDA_SYNC_CHECK, guard, align_up
│  │  ├─ timing.hpp                     # Event, CudaTimer, summarize(), (bench helpers)
│  │  ├─ data.hpp                       # Randomized initialization of matrices
│  │  └─ data.cpp                       
│  │
│  ├─ eval/
│  │  ├─ bench.h                        # Benchmarking bench_gemm(...): mean/std ms, GFLOP/s, reinit per iter
│  │  └─ verify.h                       # Verification verify_gemm(...): CPU ref C = αAB + βC, compares & reports
│  │
│  └─ bindings/
│     └─ nanobind/
│        └─ module.cpp                  # (future) Python bindings
│
├─ globals.h                            # GLOBAL_SEED and similar project-wide constants
│
├─ Git files                            # includes build/, config/config.sh, etc.
├─ (CMakeLists.txt)                     # ignored for now per your request
└─ README.md                            # (recommended: usage, build, run examples)
