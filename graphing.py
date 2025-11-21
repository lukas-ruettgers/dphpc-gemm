from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd, csv
import argparse
import sys
from enum import Enum
from io import TextIOWrapper
import subprocess
import time


KERNELS_PER_RUN = 1 # These many consecutive kernels are considered part of the same run.
WARMUP_RUNS = 0 # Skips warmup runs from the CSV.

# Cmdline arguments to pass to the program (if --bin option chosen).
# The program is executed once per item in the list.
PROGRAM_ARGS = [
    '--M 1024 --N 512 --K 256 --threadblock 32' # Example args,
]

DEFAULT_NCU_REP = 'ncu_benchmark_output.ncu-rep' # Must match ncu_benchmark.sbatch.
DEFAULT_CSV = 'ncu_benchmark_output.csv'
DEFAULT_ROOFLINE_PLOT = 'roofline_plot.png'


def get_metric_value(df: pd.DataFrame, metric_name: str) -> float:
    return float(df[df['Metric Name'] == metric_name]['Metric Value'].values[0])


@dataclass
class RunData:
    """Data for a single run."""

    performance: float # GFLOP/s
    bandwidth: float # GB/s
    runtime: float # us

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'RunData':
        sol_throughput = df[df['Section Name'] == 'GPU Speed Of Light Throughput']
        runtime = get_metric_value(sol_throughput, 'Duration') # us
        
        # NOTE: Should we add individual MUL and ADD to performance?
        sol_roofline = df[df['Section Name'] == 'GPU Speed Of Light Roofline Chart']
        sm_freq = get_metric_value(sol_roofline, 'SM Frequency') # GHz
        performance = (
            get_metric_value(sol_roofline, 'Predicated-On FFMA Operations Per Cycle') +
            get_metric_value(sol_roofline, 'Predicated-On FADD Thread Instructions Executed Per Cycle') +
            get_metric_value(sol_roofline, 'Predicated-On FMUL Thread Instructions Executed Per Cycle')
        ) * sm_freq # GFLOP/s

        bandwidth = get_metric_value(sol_roofline, 'DRAM Bandwidth') # First value is single-precision.

        return cls(
            performance = performance,
            bandwidth = bandwidth,
            runtime = runtime
        )

    
    @classmethod
    def combine(cls, data: 'list[RunData]') -> 'RunData':
        """Combines multiple RunDatas (all kernel calls belonging to the same run)."""

        total_runtime = 0
        total_gflops = 0
        total_gbs = 0

        for d in data:
            total_runtime += d.runtime
            total_gflops += d.performance * d.runtime
            total_gbs = d.bandwidth * d.runtime

        # Weighted means of performance and bandwidth.
        mean_performance = total_gflops / total_runtime
        mean_bandwidth = total_gbs / total_runtime

        return cls(
            performance = mean_performance,
            bandwidth = mean_bandwidth,
            runtime = total_runtime
        )


@dataclass
class Data(RunData):
    """Data over an entire set of runs (using median)."""

    peak_performance: float # GFLOP/s
    peak_bandwidth: float # GB/s
    arithmetic_intensity: float # FLOP/byte

    def __init__(self, dfs: list[pd.DataFrame], kernels_per_run: int, warmup_runs: int) -> None:
        # Validate grouping parameters.
        if len(dfs) % kernels_per_run != 0:
            raise Exception(f'Kernels per run ({kernels_per_run}) does not divide number of run entries in CSV ({len(dfs)})')
        run_count = len(dfs) / kernels_per_run

        if warmup_runs > run_count:
            raise Exception(f'Warmup runs ({warmup_runs}) exceeds number of runs ({run_count})')
        if warmup_runs == run_count:
            raise Exception(f'Equal number of warmup runs and total runs ({run_count})')
        
        # Skip warmup runs.
        dfs = dfs[warmup_runs * kernels_per_run : ]

        raw_runs = [RunData.from_dataframe(df) for df in dfs]
        runs = [RunData.combine(raw_runs[i : i + kernels_per_run]) for i in range(0, len(raw_runs), kernels_per_run)]

        self.performance = float(np.median([run.performance for run in runs]))
        self.bandwidth = float(np.median([run.bandwidth for run in runs]))
        self.runtime = float(np.median([run.runtime for run in runs]))

        df = dfs[0]
        sol_roofline = df[df['Section Name'] == 'GPU Speed Of Light Roofline Chart']
        sm_freq = get_metric_value(sol_roofline, 'SM Frequency') # GHz
        self.peak_performance = get_metric_value(sol_roofline, 'Theoretical Predicated-On FFMA Operations') * sm_freq # GFLOP/s
        dram_freq = get_metric_value(sol_roofline, 'DRAM Frequency') # GHz
        self.peak_bandwidth = get_metric_value(sol_roofline, 'Theoretical DRAM Bytes Accessible') * dram_freq # GB/s

        # Calculated values.
        self.arithmetic_intensity = self.performance / self.bandwidth # FLOP/byte

    
    def __str__(self) -> str:
        result = f'Peak Performance: {self.peak_performance} GFLOP/s\n'
        result += f'Peak Bandwidth: {self.peak_bandwidth} GB/s\n'
        result += f'Median Performance: {self.performance} GFLOP/s\n'
        result += f'Median Bandwidth: {self.bandwidth} GB/s\n'
        result += f'Median Runtime: {self.runtime} us\n'
        result += f'Arithmetic Intensity: {self.arithmetic_intensity} FLOP/byte\n'
        return result
        

def split_dataframe_by_id(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Splits dataframe into chunks of rows by ID"""

    ids = df['ID'].unique()
    chunks: list[pd.DataFrame] = [df[df['ID'] == id] for id in ids]
    
    return chunks


def parse_ncu_csv(csv_file: str | TextIOWrapper, kernels_per_run: int, warmup_runs: int) -> Data:
    """Parses NCU CSV file and returns a list of datapoints."""
    try:
        df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL, usecols=[
            'ID', 'Section Name','Body Item Label','Metric Name','Metric Unit','Metric Value'
        ])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    if df.empty:
        raise Exception('CSV file is empty: No kernel invocations occured')

    dfs = split_dataframe_by_id(df)
    data = Data(dfs, kernels_per_run, warmup_runs)
    return data


def plot_roofline(data: Data, title: str) -> None:
    plt.figure(figsize=(10, 6), dpi=200)
    plt.title(title, fontweight='bold', pad=20)
    
    # Set logarithmic scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Calculate ridge point
    # ridge_point = peak_compute / peak_bandwidth
    
    # Generate roofline curve
    ai_range = np.logspace(-2, 2, 500) # Arithmetic intensity range
    performance_roofline = np.minimum(data.peak_bandwidth * ai_range, data.peak_performance)
    
    # Plot roofline boundaries
    plt.plot(ai_range, performance_roofline, 'k-', linewidth=2, label='Roofline')
    plt.axhline(y=data.peak_performance, color='#707070', linestyle='--', alpha=0.7)
    
    # Plot data points
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    color = colors[1]
    plt.plot(data.arithmetic_intensity, data.performance, 'o', markersize=8, label='Kernel', color=color)
    
    # Labels and styling
    plt.xlabel('Arithmetic Intensity (FLOP/byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f8f8')
    
    # Add peak performance info on the side
    peak_info = f'Peak Performance: {data.peak_performance:.0f} GFLOP/s\nPeak Bandwidth: {data.peak_bandwidth:.0f} GB/s'
    peak_info = f'{peak_info}\nMedian Performance: {data.performance:.0f} GFLOP/s\nMedian Bandwidth: {data.bandwidth:.2f} GB/s'
    peak_info = f'{peak_info}\nArithmetic Intensity: {data.arithmetic_intensity:.2f} FLOP/byte'

    plt.annotate(peak_info, 
                xy=(1.02, 0.5), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                fontsize=10, ha='left', va='top')
    
    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(DEFAULT_ROOFLINE_PLOT, bbox_inches='tight')


def run_benchmark(executable: str) -> str:
    """Returns name of .ncu-rep file generated by NCU after running the benchmark."""

    print('Running NCU profiling...')

    cmd = ['sbatch', 'ncu_benchmark.sbatch', executable]
    subprocess.run(cmd, check=True).check_returncode()

    # Wait until `squeue` shows job is running.
    while True:
        time.sleep(3)
        print('Checking job status...')
        result = subprocess.run(['squeue'], capture_output=True, text=True)
        result.check_returncode()
        if 'nwadekar' not in result.stdout:
            break

    ncu_rep_file = DEFAULT_NCU_REP
    print(f'NCU profiling complete. Writing results to {ncu_rep_file}')
    return ncu_rep_file


def generate_csv(ncu_rep: str) -> str:
    """Generates CSV file from .ncu-rep file and returns path to CSV file."""

    csv_file = DEFAULT_CSV
    cmd = f'ncu --import {ncu_rep} --csv --print-details all --section SpeedOfLight --section SpeedOfLight_RooflineChart'.split()
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    result.check_returncode()

    print(f'Generating CSV file: {csv_file}')
    with open(csv_file, 'w') as f:
        f.write(result.stdout)
    return csv_file


class ScriptMode(Enum):
    BIN = 1
    NCU_REP = 2
    CSV = 3

def verify_args(args: argparse.Namespace) -> ScriptMode:
    """Returns the mode of execution."""

    # Priority order: bin, ncu-rep, csv.

    if args.bin is None and args.ncu_rep is None and args.csv is None:
        print('Error: Either --bin, --ncu-rep, or --csv must be provided.')
        exit(1)

    if args.bin is not None:
        return ScriptMode.BIN
    if args.ncu_rep is not None:
        return ScriptMode.NCU_REP
    return ScriptMode.CSV


def main():
    parser = argparse.ArgumentParser(description='Generate roofline plot from NSight Compute CSV')
    parser.add_argument('--bin', default=None, help='Path to the executable binary to profile')
    parser.add_argument('--ncu-rep', default=None, help='Path to the .ncu-rep file generated by NCU')
    parser.add_argument('--csv', default=None, help='Path to the CSV file containing CSV data')
    parser.add_argument('--kernels-per-run', default=KERNELS_PER_RUN, help='Number of kernel calls done per run. This value overrides the one set in the script.')
    parser.add_argument('--warmup-runs', default=WARMUP_RUNS, help='Number of warmup runs. This value overrides the one set in the script.')
    
    args = parser.parse_args()
    mode = verify_args(args)

    bin: str | None = args.bin
    ncu_rep: str | None = args.ncu_rep
    csv: str | None = args.csv

    kernels_per_run = int(args.kernels_per_run)
    warmup_runs = int(args.warmup_runs)
    
    if mode == ScriptMode.BIN:
        assert bin is not None
        
        for i, args in enumerate(PROGRAM_ARGS):
            print(f"Running benchmark {i} on binary: {bin}")
            ncu_rep = run_benchmark(bin)
            print(f"Generating CSV from NCU report: {ncu_rep}")
            csv = generate_csv(ncu_rep)
            print(f"Parsing CSV file: {csv}")
            data = parse_ncu_csv(csv, kernels_per_run, warmup_runs)
            print(f'BENCHMARK `{args}`:')
            print(data)
            print()

        plot_roofline(data, 'Roofline Analysis')

    if mode == ScriptMode.NCU_REP:
        assert ncu_rep is not None
        print(f"Generating CSV from NCU report: {ncu_rep}")
        csv = generate_csv(ncu_rep) # Proceed to CSV mode after generating CSV.

    assert csv is not None
    print(f"Parsing CSV file: {csv}")
    data = parse_ncu_csv(csv, kernels_per_run, warmup_runs)
    print(data)
    
    plot_roofline(data, 'Roofline Analysis')


if __name__ == "__main__":
    main()