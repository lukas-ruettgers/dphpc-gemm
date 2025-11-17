# Setup
First we need a virtual environment with the required packages:
```
python3 -m venv venv
source venv/bin/activate
pip install matplotlib pandas numpy
```

# Execution
- Requires a Kernel compiled with debug information. Use `profile_build.sh` as base.
- The pipeline is: binary -> `.ncu-rep` file -> `.csv` extracted -> graphed. (The generated default file names are at the top of `graphing.py`)
- You can look at `ncu_benchmark.sbatch` for the commands that are run.
- 3 modes of execution using cmdline argument for the script:
    - `--bin <binary> <args>` : Runs entire pipeline on `binary`, providing `args`.
    - `--ncu-rep <file>` : Takes `.ncu-rep` file as input and runs the rest of the pipeline.
    - `--csv <file>` : Takes a `.csv` file as input and runs the rest of the pipeline.


# Overall Intended Flow
- Benchmarking code in the library selects kernel, runs warmup runs, then runs actual runs in a loop.
- Python script runs this benchmarking binary code using `ncu` to get results.
- Script parses the results as a CSV and prints to screen and creates plots.
- Script runs this for various matrix dimensions, allows override.

NOTE: We assume one kernel invocation per run right now.
TODO: Allow multiple kernel invocations per run by having each implementation return the number of kernel invocations. (All kernel invocations in one run will be sequential in `ncu` file)