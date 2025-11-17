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

def get_metric_value(df: pd.DataFrame, metric_name: str) -> float:
    return float(df[df['Metric Name'] == metric_name]['Metric Value'].values[0])


@dataclass
class RunData:
    """Data for a single run."""

    performance: float # GFLOP/s
    bandwidth: float # GB/s
    runtime: float # us

    def __init__(self, df: pd.DataFrame) -> None:
        sol_throughput = df[df['Section Name'] == 'GPU Speed Of Light Throughput']
        self.runtime = get_metric_value(sol_throughput, 'Duration') # us
        
        # NOTE: Should we add individual MUL and ADD to performance?
        sol_roofline = df[df['Section Name'] == 'GPU Speed Of Light Roofline Chart']
        sm_freq = get_metric_value(sol_roofline, 'SM Frequency') # GHz
        self.performance = (
            get_metric_value(sol_roofline, 'Predicated-On FFMA Operations Per Cycle') +
            get_metric_value(sol_roofline, 'Predicated-On FADD Thread Instructions Executed Per Cycle') +
            get_metric_value(sol_roofline, 'Predicated-On FMUL Thread Instructions Executed Per Cycle')
        ) * sm_freq # GFLOP/s
        self.bandwidth = get_metric_value(sol_roofline, 'DRAM Bandwidth') # First value is single-precision.


@dataclass
class Data(RunData):
    """Data over an entire set of runs (using median)."""

    peak_performance: float # GFLOP/s
    peak_bandwidth: float # GB/s
    arithmetic_intensity: float # FLOP/byte

    def __init__(self, dfs: list[pd.DataFrame]) -> None:
        runs = [RunData(df) for df in dfs]

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


def parse_ncu_csv(csv_file: str | TextIOWrapper) -> Data:
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
    data = Data(dfs)
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
    plt.savefig('roofline_plot.png', bbox_inches='tight')


def run_benchmark(cmd: list[str]) -> str:
    """Returns name of .ncu-rep file generated by NCU after running the benchmark."""

    print('Running NCU profiling...')

    cmd = ['sbatch', 'ncu_benchmark.sbatch'] + cmd
    subprocess.run(cmd, check=True).check_returncode()

    # Wait until `squeue` shows job is running.
    while True:
        time.sleep(3)
        print('Checking job status...')
        result = subprocess.run(['squeue'], capture_output=True, text=True)
        result.check_returncode()
        if 'nwadekar' not in result.stdout:
            break

    ncu_rep_file = 'ncu_benchmark_output.ncu-rep' # Must match ncu.sbatch.
    print(f'NCU profiling complete. Writing results to {ncu_rep_file}')
    return ncu_rep_file


def generate_csv(ncu_rep: str) -> str:
    """Generates CSV file from .ncu-rep file and returns path to CSV file."""

    csv_file = 'ncu_benchmark_output.csv'
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
    parser.add_argument('--bin', nargs='+', default=None, help='Path to the executable binary to profile and cmdline args')
    parser.add_argument('--ncu-rep', default=None, help='Path to the .ncu-rep file generated by NCU')
    parser.add_argument('--csv', default=None, help='Path to the CSV file containing CSV data')
    parser.add_argument('--title', default='GPU Roofline Analysis', help='Plot title')
    
    args = parser.parse_args()
    mode = verify_args(args)

    bin: list[str] | None = args.bin
    ncu_rep: str | None = args.ncu_rep
    csv: str | None = args.csv
    
    if mode == ScriptMode.BIN:
        assert bin is not None
        print(f"Running benchmark on binary: {bin}")
        ncu_rep = run_benchmark(bin)
        mode = ScriptMode.NCU_REP  # Proceed to NCU_REP mode after running benchmark.

    if mode == ScriptMode.NCU_REP:
        assert ncu_rep is not None
        print(f"Generating CSV from NCU report: {ncu_rep}")
        csv = generate_csv(ncu_rep)

    assert csv is not None
    print(f"Parsing CSV file: {csv}")
    data = parse_ncu_csv(csv)
    print(data)
    
    plot_roofline(data, args.title)


if __name__ == "__main__":
    main()