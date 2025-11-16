- Benchmarking code in the library selects kernel, runs warmup runs, then runs actual runs in a loop.
- Python script runs this benchmarking binary code using `ncu` to get results.
- Script parses the results as a CSV and prints to screen and creates plots.
- Script runs this for various matrix dimensions, allows override.

NOTE: We assume one kernel invocation per run right now.
TODO: Allow multiple kernel invocations per run by having each implementation return the number of kernel invocations. (All kernel invocations in one run will be sequential in `ncu` file)