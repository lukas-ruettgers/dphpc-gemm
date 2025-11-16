from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd, csv
import argparse
import sys

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
        self.runtime = get_metric_value(sol_throughput, 'Duration')
        
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
        return result
        

def split_dataframe(df: pd.DataFrame, column: str, value: str) -> list[pd.DataFrame]:
    """Splits dataframe into chunks of rows with delimiter being rows where column == value"""

    # Find indices where the split condition occurs
    split_indices = df[df[column] == value].index.tolist()
    
    # Add start index
    all_indices = [-1] + split_indices
    
    # Create chunks
    chunks: list[pd.DataFrame] = []
    for i in range(len(all_indices) - 1):
        start = all_indices[i] + 1 # Skip the delimiter row
        end = all_indices[i + 1]
        chunk = df.iloc[start:end]
        chunks.append(chunk)
    
    return chunks


def parse_ncu_csv(csv_file: str) -> Data:
    """Parses NCU CSV file and returns a list of datapoints."""
    try:
        df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL, usecols=[
            'Section Name','Body Item Label','Metric Name','Metric Unit','Metric Value'
        ])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    if df.empty:
        raise Exception('CSV file is empty: No kernel invocations occured')

    dfs = split_dataframe(df, 'Section Name', 'SpeedOfLight_RooflineChart')
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


def main():
    parser = argparse.ArgumentParser(description='Generate roofline plot from NSight Compute CSV')
    parser.add_argument('csv_file', help='Path to the NSight Compute CSV file')
    parser.add_argument('--title', default='GPU Roofline Analysis', help='Plot title')
    
    args = parser.parse_args()
    
    # TODO: Run kernel.
    
    print(f"Parsing CSV file: {args.csv_file}")
    data = parse_ncu_csv(args.csv_file)
    print(data)
    
    plot_roofline(data, args.title)


if __name__ == "__main__":
    main()