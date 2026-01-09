import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

PEAK_TFLOPS = 23.69
PEAK_BANDWIDTH_GBPS = 448

# Problem sizes (M = N = K)
DEFAULT_SIZES = [512, 1024, 2048, 4096, 8192]
DEFAULT_TOTAL_FLOPS = [2 * (size ** 3) for size in DEFAULT_SIZES]
DEFAULT_TOTAL_BYTES = [3 * (size ** 2) * 4 for size in DEFAULT_SIZES]
DEFAULT_INTENSITIES = [f / b for f, b in zip(DEFAULT_TOTAL_FLOPS, DEFAULT_TOTAL_BYTES)]

class Data:
    label: str
    tflops: list[float]
    tflops_err: list[float]
    fmt: str

    def __init__(self, label: str, fmt: str, gflops: list[float], gflops_err: list[float]) -> None:
        self.label = label
        self.tflops = [gf / 1000.0 for gf in gflops]
        self.tflops_err = [err / 1000.0 for err in gflops_err]
        self.fmt = fmt

    def plt_errorbar(self, sizes: list[int]) -> None:
        plt.errorbar(
            sizes, self.tflops, yerr=self.tflops_err,
            fmt=self.fmt, capsize=8, elinewidth=2, capthick=2,
            linewidth=1.5, markersize=6,
            label=self.label
        )


CUTLASS_BASELINE = Data(
    "[Baseline] CUTLASS", ':g',
    [6109.87, 14811.4, 15426.3, 16059.1, 15866.9],
    [35.4717, 20.7781, 23.9136, 99.6917, 12.8647])


@dataclass
class Plot:
    fig: str
    title: str
    data: list[Data]
    sizes: list[int]

    def __init__(self, fig: str, title: str, data: list[Data], sizes: list[int] = DEFAULT_SIZES) -> None:
        self.fig = fig
        self.title = title
        self.data = data
        self.sizes = sizes


    def plot(self) -> None:
        plt.axhline(
            y=PEAK_TFLOPS, 
            color='red', 
            linestyle=(0, (5, 2)), # "Small-dashed" pattern: 5pt dash, 2pt space
            linewidth=1.2,
            zorder=2
        )

        plt.text(
            x=sum(self.sizes) / len(self.sizes), 
            y=PEAK_TFLOPS * 1.005, # above the line
            s=f'RTX 5060 Ti Peak: {PEAK_TFLOPS} TFLOP/s',
            color='red',
            fontsize=8,
            va='bottom',
            ha='left'
        )

        for d in self.data:
            d.plt_errorbar(self.sizes)

        plt.xscale('log', base=2)

        plt.xticks(self.sizes, labels=[str(s) for s in self.sizes])

        plt.xlabel('GEMM Shape (M = N = K)', fontsize=11)
        plt.ylabel('Achieved TFLOP/s', fontsize=11)
        plt.title(
            self.title,
            fontsize=12
        )

        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

        plt.legend(frameon=False, fontsize=8)

        plt.tight_layout()
        plt.savefig(
            self.fig,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


    def plot_roofline(self) -> None:
        # Arithmetic Intensity range
        intensities = np.logspace(-1, 4, 100)
        
        bw_term = (PEAK_BANDWIDTH_GBPS / 1000.0) * intensities
        roofline = np.minimum(bw_term, PEAK_TFLOPS)
        
        plt.plot(intensities, roofline, color='black', linewidth=2, label='Roofline', zorder=3)
        
        plt.axhline(y=PEAK_TFLOPS, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ridge_point = PEAK_TFLOPS / (PEAK_BANDWIDTH_GBPS / 1000.0)
        plt.axvline(x=ridge_point, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        
        for tflops, intensity, size in zip(self.data[0].tflops, DEFAULT_INTENSITIES, DEFAULT_SIZES):
            plt.scatter(intensity, tflops, label=f'M=N=K = {size}', zorder=4)

        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        
        plt.xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=11)
        plt.ylabel('Achieved Performance (TFLOP/s)', fontsize=11)
        plt.title(f'Roofline Analysis: {self.title}', fontsize=12)
        
        plt.legend(frameon=False, fontsize=8)
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(self.fig, dpi=300, bbox_inches="tight")
        plt.close()


##### ADD ALL MEASURED VALUES BELOW #####

# -------- Vec --------
# TB_M, TB_N, TB_K = 16, 16, 16

vec = Plot(
    fig='gemm_vec.pdf',
    title='Effect of Vectorised Loads',
    data=[
        CUTLASS_BASELINE,
        Data("Non-blocked Matrices", 'o-',
            [1585.75, 1730.05, 1760.1, 1620.62, 1628.09],
            [2.23218, 0.878799, 0.109204, 0.228695, 0.0174399]),
        Data("Blocked Matrices", 's--',
            [2007.89, 1862.98, 1874.28, 1950.93, 1965.93],
            [3.5052, 1.0459, 0.131876, 0.0581686, 1.46353]),
    ]
)

# -------- Vec 1D --------
# TB_M, TB_N, TB_K = 64, 64, 8
# TM = 8 (1d block tiling)

vec_1d = Plot(
    fig='gemm_vec_1d.pdf',
    title='Effect of 1D block tiling',
    data=[
        CUTLASS_BASELINE,
        Data("Non-blocked Matrices", 'o-',
            [3254.58, 3738.69, 4917.98, 3366.41, 3294.53],
            [23.5918, 2.46432, 3.96977, 1.04084, 1.67082]),
        Data("Blocked Matrices", 's--',
            [3468.75, 3989.62, 5227.46, 4204.31, 4235.29],
            [6.8669, 1.94869, 3.60132, 0.985214, 1.45182]),
    ]
)

# -------- Vec 2D --------
# TB_M, TB_N, TB_K = 128, 128, 16
# TM, TN = 8, 8 (2d block tiling)

vec_2d = Plot(
    fig='gemm_vec_2d.pdf',
    title='Effect of 2D block tiling',
    data=[
        CUTLASS_BASELINE,
        Data("Non-blocked Matrices", 'o-',
            [2880.07, 8169.9, 9039.25, 8629.42, 9653.5],
            [15.2099, 25.6378, 8.3107, 12.0924, 47.2688]),
        Data("Blocked Matrices", 's--',
            [2798.14, 7933.35, 9085.32, 9209.44, 9412.06],
            [6.81251, 9.67044, 15.4387, 5.27224, 3.17451]),
    ]
)

# -------- Vec Warp --------
# TB_M, TB_N, TB_K = 128, 128, 16
# TM, TN = 8, 4 (2d block tiling)
# W_M, W_N = 64, 64
# WNITER = 4

vec_warp = Plot(
    fig='gemm_vec_warp.pdf',
    title='Effect of Warp-tiling',
    data=[
        CUTLASS_BASELINE,
        Data("Non-blocked Matrices", 'o-',
            [3287.43, 10190.4, 12118.6, 13570.6, 13723.5],
            [8.98124, 9.6063, 14.3264, 38.5838, 15.7164]),
        Data("Blocked Matrices", 's--',
            [4873.3, 12610.5, 13854.8, 14802.1, 14851.6],
            [21.1037, 30.8059, 46.7267, 55.337, 11.4763]),
    ]
)

# -------- Vec Double Buf --------
# TB_M, TB_N, TB_K = 128, 128, 16
# TM, TN = 8, 4 (2d block tiling)
# W_M, W_N = 64, 64
# WNITER = 4

vec_double_buf = Plot(
    fig='gemm_vec_double_buf.pdf',
    title='Effect of Double-buffering',
    data=[
        CUTLASS_BASELINE,
        Data("Blocked Matrices", 's--C1',
            [4828.37, 14584.6, 15045.7, 15656.8, 15604.8],
            [17.1998, 28.6328, 28.2549, 71.6012, 11.4836]),
        Data("Non-blocked Matrices (Sync Loads)", 'o-C0',
            [2796.37, 8581.78, 10366.9, 11586.5, 11249.6],
            [9.80297, 37.1253, 20.3312, 26.2581, 6.48886]),
        Data("Non-blocked Matrices (Async Loads)", 'o-m',
            [1617.46, 5614.1, 5497, 5887.91, 6199.02],
            [3.12216, 3.46352, 2.81872, 1.73387, 0.291235]),
    ]
)

# -------- Vec Double Buf Roofline --------
roofline = Plot(
    fig='gemm_vec_double_buf_roofline.pdf',
    title='Roofline Plot',
    data=[
        Data("Blocked Matrices", 's--C1',
            [4828.37, 14584.6, 15045.7, 15656.8, 15604.8],
            [17.1998, 28.6328, 28.2549, 71.6012, 11.4836]),
    ]
)

# -------- Plot --------

# vec.plot()
# vec_1d.plot()
# vec_2d.plot()
# vec_warp.plot()
# vec_double_buf.plot()
# roofline.plot_roofline()

# Error bars indicate 95% confidence intervals over 50 runs.