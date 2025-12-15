import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

# ====================
# GLOBAL CONFIGURATION
# ====================

USE_FILE = False  # Set to True to read points from a file
FILENAME = 'data_points.txt'  # File containing points if USE_FILE is True

# Graph appearance
GRAPH_TITLE = "Line Graph with Error Bars"
X_AXIS_LABEL = "X Axis"
Y_AXIS_LABEL = "Y Axis"
FIG_SIZE = (10, 6)  # (width, height) in inches
DPI = 100  # Resolution of the figure

# Line style
LINE_COLOR = "blue"
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
OUTPUT_FILENAME = "line_graph.png"
OUTPUT_DPI = 300  # Resolution for saved file

# Statistics settings
USE_MEDIAN = True  # If False, uses mean instead of median
ERROR_BAR_SIGMA = 1.0  # Multiplier for standard deviation (1.0 = 1 std dev, 2.0 = 2 std dev, etc.)

# ====================
# PLOTTING FUNCTION
# ====================

def plot_line_graph(points: list[tuple[float, float]]):
    """
    Plot a line graph using the provided points and global configuration.
    Handles multiple y values for the same x value by calculating median and standard deviation.
    
    Parameters:
    -----------
    points : List[Tuple[float, float]]
        List of (x, y) tuples to plot. Multiple y values for same x are aggregated.
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Group y values by x coordinate
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
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # Set background color
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Plot the line with markers
    line_plot = ax.plot(x_values, y_central, 
                        color=LINE_COLOR, 
                        linewidth=LINE_WIDTH, 
                        linestyle=LINE_STYLE,
                        marker=MARKER_STYLE,
                        markersize=MARKER_SIZE,
                        markerfacecolor=MARKER_COLOR,
                        markeredgecolor=MARKER_EDGE_COLOR,
                        markeredgewidth=MARKER_EDGE_WIDTH,
                        label=f"{LEGEND_LABEL} ({central_label})" if SHOW_LEGEND else None)
    
    # Add error bars if enabled
    if SHOW_ERROR_BARS and len(y_stdevs) > 0 and any(stdev > 0 for stdev in y_stdevs):
        error_amounts = [stdev * ERROR_BAR_SIGMA for stdev in y_stdevs]
        
        errorbar_plot = ax.errorbar(x_values, y_central, 
                                    yerr=error_amounts,
                                    fmt='none',  # Don't plot markers (we already have them)
                                    ecolor=ERROR_BAR_COLOR,
                                    elinewidth=ERROR_BAR_WIDTH,
                                    capsize=ERROR_BAR_CAP_SIZE,
                                    alpha=ERROR_BAR_ALPHA,
                                    label=f"±{ERROR_BAR_SIGMA} Std Dev" if SHOW_LEGEND else None)
    
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
    
    # Add statistics summary to the plot
    total_points = len(points)
    unique_x = len(grouped_data)
    avg_y_per_x = total_points / unique_x if unique_x > 0 else 0
    
    stats_text = (f"Statistics Summary:\n"
                  f"Total points: {total_points}\n"
                  f"Unique x-values: {unique_x}\n"
                  f"Avg y per x: {avg_y_per_x:.2f}")
    
    # Add text box with statistics
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout
    # plt.tight_layout()
    
    # Save figure
    plt.savefig(OUTPUT_FILENAME, dpi=OUTPUT_DPI, bbox_inches='tight')
    print(f"Figure saved as {OUTPUT_FILENAME}")
    
    # Print statistics to console
    print("\n=== Statistics Summary ===")
    print(f"Total data points: {total_points}")
    print(f"Unique x-values: {unique_x}")
    print(f"Average y-values per x: {avg_y_per_x:.2f}")
    print(f"Using {central_label.lower()} for central tendency")
    print(f"Error bars show ±{ERROR_BAR_SIGMA} standard deviations")
    
    return fig, ax


def main():
    points = []
    if USE_FILE:
        # Read points from file
        with open(FILENAME, 'r') as f:
            for line in f:
                x_str, y_str = line.strip().split(',')
                points.append((float(x_str), float(y_str)))
    else:
        # Use hardcoded example points
        points = [
            (0, 1.0), (0, 1.2), (0, 0.9),
            (1, 2.1), (1, 2.3), (1, 1.8),
            (2, 3.0), (2, 3.2), (2, 2.9),
            (3, 4.1), (3, 4.3), (3, 3.8),
            (4, 5.0), (4, 5.2), (4, 4.9),
        ]
    
    plot_line_graph(points)


if __name__ == "__main__":
    main()