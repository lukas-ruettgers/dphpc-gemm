import matplotlib.pyplot as plt
from collections import defaultdict
import statistics
import os
import numpy as np

# ====================
# GLOBAL CONFIGURATION
# ====================

USE_FILE = False  # Set to True to read points from a file
FILENAME = 'data_points.txt'  # File containing points if USE_FILE is True

OUTPUT_FILENAME = os.path.join(os.getcwd(), "graphing/graphs/matrix_size.png")

# Graph appearance
GRAPH_TITLE = "Runtime with varying Square Matrix Size"
X_AXIS_LABEL = "X Axis"
Y_AXIS_LABEL = "Y Axis"
FIG_SIZE = (10, 6)  # (width, height) in inches
DPI = 100  # Resolution of the figure

# Line style
LINE_COLORS = ["blue", "red"]
LINE_WIDTH = 2
LINE_STYLE = "-"  # "-" for solid, "--" for dashed, ":" for dotted, "-." for dash-dot
MARKER_STYLE = "o"  # "o" for circles, "s" for squares, "^" for triangles, etc.
MARKER_SIZE = 8
MARKER_COLOR = "red"
MARKER_EDGE_COLOR = "black"
MARKER_EDGE_WIDTH = 1

# Error bar style
ERROR_BAR_COLOR = "black"
ERROR_BAR_WIDTH = 2
ERROR_BAR_CAP_SIZE = 5
SHOW_ERROR_BARS = True
ERROR_BAR_ALPHA = 0.7
ERROR_BAR_STYLE = "-"  # "-" for solid, "--" for dashed

# Grid and background
SHOW_GRID = True
GRID_STYLE = "--"
GRID_COLOR = "gray"
GRID_ALPHA = 0.3
BACKGROUND_COLOR = "white"

# Axis limits (set to None for auto-scaling)
X_LIMITS = None  # (min, max) or None
Y_LIMITS = None  # (min, max) or None

# Axis ticks
X_TICK_COUNT = 10  # Approximate number of ticks (set to None for auto)
Y_TICK_COUNT = 10  # Approximate number of ticks (set to None for auto)
SHOW_TICK_LABELS = True

# Legend
SHOW_LEGEND = True
LEGEND_LABEL = "Median Values"
LEGEND_LOCATION = "best"  # "upper left", "upper right", "lower left", "lower right", "best"
LEGEND_FRAME_ALPHA = 0.9

# Font sizes
TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 12

# Output settings
OUTPUT_DPI = 300  # Resolution for saved file

# Statistics settings
USE_MEDIAN = False  # If False, uses mean instead of median
ERROR_BAR_SIGMA = 1.0  # Multiplier for standard deviation (1.0 = 1 std dev, 2.0 = 2 std dev, etc.)

# ====================
# PLOTTING FUNCTION
# ====================

def plot_line_graphs(points_lists: list[list[tuple[float, float]]], line_names: list[str]):
    """
    Plot multiple line graphs using the provided points lists and global configuration.
    Handles multiple y values for the same x value by calculating median and standard deviation.
    
    Parameters:
    -----------
    points_lists : List[List[Tuple[float, float]]]
        List of lists, each containing (x, y) tuples for one line
    line_names : List[str]
        Names for each line to use in the legend
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    POWERS_OF_2_TICKS = [128, 256, 512, 1024, 2048, 4096, 8192]  # Example powers of 2

    # Set x-axis to log base 2 scale for powers of 2 display
    ax.set_xscale('log', base=2)
    ax.set_xlim(left=min(POWERS_OF_2_TICKS) // 2, right=max(POWERS_OF_2_TICKS) * 2)

    # If you want to set custom x-ticks at powers of 2:
    if hasattr(POWERS_OF_2_TICKS, '__len__') and len(POWERS_OF_2_TICKS) > 0:
        ax.set_xticks(POWERS_OF_2_TICKS)
        ax.set_xticklabels([str(p) for p in POWERS_OF_2_TICKS])
    
    # Set background color
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Colors for different lines
    line_colors = LINE_COLORS if hasattr(LINE_COLORS, '__len__') else plt.cm.tab10.colors
    
    total_points = 0
    unique_x_total = 0
    
    # Plot each line
    for idx, points in enumerate(points_lists):
        # Group y values by x coordinate for this line
        grouped_data = defaultdict(list)
        for x, y in points:
            grouped_data[x].append(y)
        
        # Calculate statistics for each x value
        x_values = []
        y_medians = []
        y_stdevs = []
        y_means = []
        
        for x in sorted(grouped_data.keys()):
            y_list = grouped_data[x]
            
            if len(y_list) > 0:
                x_values.append(x)
                
                if USE_MEDIAN:
                    # Calculate median
                    try:
                        median_val = statistics.median(y_list)
                    except statistics.StatisticsError:
                        median_val = float('nan')
                    y_medians.append(median_val)
                else:
                    # Calculate mean
                    y_means.append(statistics.mean(y_list))
                
                # Calculate standard deviation (only if we have at least 2 values)
                if len(y_list) > 1:
                    stdev_val = statistics.stdev(y_list)
                else:
                    stdev_val = 0  # No standard deviation for single value
                y_stdevs.append(stdev_val)
        
        # Choose which central tendency measure to use
        if USE_MEDIAN:
            y_central = y_medians
            central_label = "Median"
        else:
            y_central = y_means
            central_label = "Mean"
        
        # Get color for this line
        line_color = line_colors[idx % len(line_colors)]
        
        # Plot the line with markers
        line_plot = ax.plot(x_values, y_central, 
                            color=line_color, 
                            linewidth=LINE_WIDTH, 
                            linestyle=LINE_STYLE,
                            marker=MARKER_STYLE,
                            markersize=MARKER_SIZE,
                            markerfacecolor=MARKER_COLOR,
                            markeredgecolor=MARKER_EDGE_COLOR,
                            markeredgewidth=MARKER_EDGE_WIDTH,
                            label=line_names[idx] if SHOW_LEGEND else None)
        
        # Add error bars if enabled
        if SHOW_ERROR_BARS and len(y_stdevs) > 0 and any(stdev > 0 for stdev in y_stdevs):
            error_amounts = [stdev * ERROR_BAR_SIGMA for stdev in y_stdevs]
            
            errorbar_plot = ax.errorbar(x_values, y_central, 
                                        yerr=error_amounts,
                                        fmt='none',  # Don't plot markers (we already have them)
                                        ecolor=line_color,
                                        elinewidth=ERROR_BAR_WIDTH,
                                        capsize=ERROR_BAR_CAP_SIZE,
                                        alpha=ERROR_BAR_ALPHA,
                                        label=f"{line_names[idx]} ±{ERROR_BAR_SIGMA}σ" if SHOW_LEGEND else None)
        
        # Update statistics
        total_points += len(points)
        unique_x_total += len(grouped_data)
    
    # Set title and labels
    ax.set_title(GRAPH_TITLE, fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=10)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=10)
    
    # Set axis limits if specified
    if X_LIMITS is not None:
        ax.set_xlim(X_LIMITS)
    if Y_LIMITS is not None:
        ax.set_ylim(Y_LIMITS)
    
    # Configure ticks
    if X_TICK_COUNT is not None:
        ax.locator_params(axis='x', nbins=X_TICK_COUNT)
    if Y_TICK_COUNT is not None:
        ax.locator_params(axis='y', nbins=Y_TICK_COUNT)
    
    # Set tick label font size
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONT_SIZE)
    
    # Configure grid
    if SHOW_GRID:
        ax.grid(True, linestyle=GRID_STYLE, color=GRID_COLOR, alpha=GRID_ALPHA)
    
    # Add legend if enabled
    if SHOW_LEGEND:
        legend = ax.legend(loc=LEGEND_LOCATION, fontsize=LEGEND_FONT_SIZE)
        legend.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
    
    # Calculate overall statistics
    avg_y_per_x = total_points / unique_x_total if unique_x_total > 0 else 0
    
    stats_text = (f"Statistics Summary:\n"
                  f"Total points: {total_points}\n"
                  f"Total lines: {len(points_lists)}\n"
                  f"Unique x-values per line: ~{unique_x_total/len(points_lists):.0f} avg")
    
    # Add text box with statistics
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(OUTPUT_FILENAME, dpi=OUTPUT_DPI, bbox_inches='tight')
    print(f"Figure saved as {OUTPUT_FILENAME}")
    
    # Print statistics to console
    print("\n=== Statistics Summary ===")
    print(f"Total data points: {total_points}")
    print(f"Number of lines: {len(points_lists)}")
    print(f"Average unique x-values per line: {unique_x_total/len(points_lists):.1f}")
    print(f"Using {'median' if USE_MEDIAN else 'mean'} for central tendency")
    if SHOW_ERROR_BARS:
        print(f"Error bars show ±{ERROR_BAR_SIGMA} standard deviations")
    
    return fig, ax


def parse_file(filename, delimiter="=========="):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    lists = []
    current_list = []
    
    for line in lines:
        line = line.strip()
        if line == delimiter:
            if current_list:
                lists.append(current_list)
            current_list = []
            continue
        if line:
            num = float(line)
            current_list.append(num)
    
    if current_list:
        lists.append(current_list)
    
    return lists


def main():
    unblocked_points = []
    blocked_points = []

    for size in [128, 256, 512, 1024, 2048, 4096, 8192]:
        filename = f'graphing/data/size/m{size}_n{size}_k{size}.txt'
        data: list[list[float]] = parse_file(filename)
        unblocked_points.extend((size, d) for d in data[0])
        blocked_points.extend((size, d) for d in data[1])

    
    plot_line_graphs([unblocked_points, blocked_points], ["Non-blocked GEMM", "Blocked GEMM"])

if __name__ == "__main__":
    main()