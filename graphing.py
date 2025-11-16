import matplotlib.pyplot as plt
import numpy as np
import pandas as pd, csv
import argparse
import sys

def parse_ncu_csv(csv_file):
    """
    Parse NSight Compute CSV file to extract roofline data
    Returns: (peak_compute, peak_bandwidth, actual_compute, actual_intensity)
    """
    try:
        df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    # Relevant columns: "Section Name","Body Item Label","Metric Name","Metric Unit","Metric Value"
    
    # Filter for single precision data
    single_precision_data = df[df['Body Item Label'] == 'Single Precision Roofline']
    achieved_data = df[df['Body Item Label'] == 'Single Precision Achieved Value']
    
    if single_precision_data.empty or achieved_data.empty:
        print("Error: Could not find required Single Precision data in CSV")
        sys.exit(1)
    
    # Extract peak values (should be constant across entries)
    peak_compute_row = single_precision_data[
        single_precision_data['Metric Name'] == 'Theoretical Predicated-On FFMA Operations'
    ]
    peak_sm_freq_row = single_precision_data[
        single_precision_data['Metric Name'] == 'SM Frequency'
    ]
    
    peak_dram_bw_row = single_precision_data[
        single_precision_data['Metric Name'] == 'Theoretical DRAM Bytes Accessible'
    ]
    peak_dram_freq_row = single_precision_data[
        single_precision_data['Metric Name'] == 'DRAM Frequency'
    ]
    
    if peak_compute_row.empty or peak_sm_freq_row.empty or peak_dram_bw_row.empty or peak_dram_freq_row.empty:
        print("Error: Could not find all required peak performance metrics")
        sys.exit(1)
    
    # Get peak values (take first occurrence since they should be constant)
    peak_ffma_ops = float(peak_compute_row.iloc[0]['Metric Value'])  # inst
    peak_sm_freq = float(peak_sm_freq_row.iloc[0]['Metric Value'])   # GHz
    peak_dram_bytes_per_cycle = float(peak_dram_bw_row.iloc[0]['Metric Value'])  # byte/cycle
    peak_dram_freq = float(peak_dram_freq_row.iloc[0]['Metric Value'])  # GHz
    
    # Calculate peak compute performance (GFLOP/s)
    # FFMA operations per cycle * SM frequency = GFLOP/s
    peak_compute = peak_ffma_ops * peak_sm_freq
    
    # Calculate peak memory bandwidth (GB/s)
    # bytes per cycle * DRAM frequency = GB/s
    peak_bandwidth = (peak_dram_bytes_per_cycle * peak_dram_freq)
    
    # Extract achieved performance values
    achieved_perf_rows = achieved_data[
        achieved_data['Metric Name'] == 'Predicated-On FFMA Operations Per Cycle'
    ]
    achieved_sm_freq_rows = achieved_data[
        achieved_data['Metric Name'] == 'SM Frequency'
    ]
    achieved_bw_rows = achieved_data[
        achieved_data['Metric Name'] == 'DRAM Bandwidth'
    ]
    
    if achieved_perf_rows.empty or achieved_bw_rows.empty:
        print("Error: Could not find achieved performance metrics")
        sys.exit(1)

    # GFLOP/s
    achieved_perfs = achieved_perf_rows['Metric Value'].astype(float).values * achieved_sm_freq_rows['Metric Value'].astype(float).values
    # GB/s
    achieved_bws = achieved_bw_rows['Metric Value'].astype(float).values
    
    # FLOP/byte
    achieved_intensities = achieved_perfs / achieved_bws

    achieved_perf_median = np.median(achieved_perfs)
    achieved_intensity_median = np.median(achieved_intensities)
    
    return peak_compute, peak_bandwidth, achieved_perf_median, achieved_intensity_median


def plot_roofline(peak_compute, peak_bandwidth, actual_compute, actual_intensity, title="Roofline Plot"):
    """
    Simple roofline plot with clean gray color scheme
    
    Parameters:
    - peak_compute: Peak compute performance (GFLOP/s)
    - peak_bandwidth: Peak memory bandwidth (GB/s)
    - data_points: List of tuples (arithmetic_intensity, performance, label)
    - title: Plot title
    """
    
    plt.figure(figsize=(10, 6), dpi=200)
    plt.title(title, fontweight='bold', pad=20)
    
    # Set logarithmic scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Calculate ridge point
    ridge_point = peak_compute / peak_bandwidth
    
    # Generate roofline curve
    ai_range = np.logspace(-2, 2, 500)
    performance_roofline = np.minimum(peak_bandwidth * ai_range, peak_compute)
    
    # Plot roofline boundaries
    plt.plot(ai_range, performance_roofline, 'k-', linewidth=2, label='Roofline')
    plt.axhline(y=peak_compute, color='#707070', linestyle='--', alpha=0.7)
    
    # Plot data points
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    color = colors[1]
    plt.plot(actual_intensity, actual_compute, 'o', markersize=8, label='Kernel', color=color)
    
    # Labels and styling
    plt.xlabel('Arithmetic Intensity (FLOP/byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f8f8')
    
    # Add peak performance info on the side
    peak_info = f'Peak Compute: {peak_compute:.0f} GFLOP/s\nPeak Bandwidth: {peak_bandwidth:.0f} GB/s'
    peak_info = f'{peak_info}\nActual Compute: {actual_compute:.0f} GFLOP/s\nActual Intensity: {actual_intensity:.2f} FLOP/byte'
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
    
    print(f"Parsing CSV file: {args.csv_file}")
    peak_compute, peak_bandwidth, actual_compute, actual_intensity = parse_ncu_csv(args.csv_file)
    
    print(f"Peak Compute: {peak_compute:.2f} GFLOP/s")
    print(f"Peak Bandwidth: {peak_bandwidth:.2f} GB/s")
    print(f"Actual Compute: {actual_compute:.2f} GFLOP/s")
    print(f"Actual Intensity: {actual_intensity:.2f} FLOP/byte")
    
    plot_roofline(peak_compute, peak_bandwidth, actual_compute, actual_intensity, args.title)


if __name__ == "__main__":
    main()