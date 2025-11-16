import matplotlib.pyplot as plt
import numpy as np

def plot_roofline(peak_compute, peak_bandwidth, data_points, title="Roofline Plot"):
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
    plt.axhline(y=peak_compute, color='#707070', linestyle='--', alpha=0.7, 
                label=f'Peak Compute: {peak_compute:.0f} GFLOP/s')
    
    # Plot data points
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (ai, perf, label) in enumerate(data_points):
        color = colors[i % len(colors)]
        plt.plot(ai, perf, 'o', markersize=8, label=label, color=color)
    
    # Labels and styling
    plt.xlabel('Arithmetic Intensity (FLOP/byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.grid(True, alpha=0.3)
    plt.gca().set_facecolor('#f8f8f8')
    
    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('roofline_plot.png')


if __name__ == "__main__":
    # Your GPU specifications
    PEAK_COMPUTE = 22118.4  # GFLOP/s (from your NSight data)
    PEAK_BANDWIDTH = 441.28  # GB/s (from your NSight data)
    
    # Your data points: (arithmetic_intensity, performance, label)
    MY_DATA_POINTS = [
        (22.66, 2233, "My Kernel"),
        (15.0, 1800, "Kernel A"),
        (30.0, 2500, "Kernel B"),
    ]
    
    plot_roofline(PEAK_COMPUTE, PEAK_BANDWIDTH, MY_DATA_POINTS, "GPU Roofline Analysis")