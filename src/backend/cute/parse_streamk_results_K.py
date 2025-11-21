#!/usr/bin/env python3
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_results(filename):
    """
    Parse the StreamK sweep output file and extract per-BN runtimes and GFLOPs
    for each of the five kernel configurations.

    Returns
    -------
    bn_values : np.ndarray
        Array of BN indices (1, 2, 3, ...).
    n_values : np.ndarray
        Array of N values (usually 128 * BN).
    runtimes : dict[str, np.ndarray]
        Mapping from kernel label -> runtime array [ms] over BN.
    gflops : dict[str, np.ndarray]
        Mapping from kernel label -> GFLOPs array over BN.
    """

    # Kernel names exactly as they appear in the log
    kernel_patterns = [
        ("Basic data-parallel GEMM", "basic_dp"),
        ("StreamK GEMM with default load-balancing", "streamk_default"),
        ("StreamK emulating basic data-parallel GEMM", "streamk_emul_dp"),
        ("Basic split-K GEMM with tile-splitting factor 2", "basic_splitk"),
        ("StreamK emulating Split-K GEMM with tile-splitting factor 2", "streamk_emul_splitk"),
    ]

    # Order for plotting / arrays
    kernel_order = [k[1] for k in kernel_patterns]

    # Initialize containers
    bn_list = []
    n_list = []

    runtimes = {key: [] for key in kernel_order}
    gflops = {key: [] for key in kernel_order}

    # Regex to detect BN header lines:
    # --- BN=1, n=128 ---
    re_bk = re.compile(r"^---\s*BK=(\d+),\s*n=(\d+)\s*---")

    current_bn = None
    current_kernel_key = None
    temp_runtime = None

    with open(filename, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Detect start of a BN block
            m_bn = re_bk.match(line)
            if m_bn:
                current_bn = int(m_bn.group(1))
                current_n = int(m_bn.group(2))
                bn_list.append(current_bn)
                n_list.append(current_n)
                current_kernel_key = None
                temp_runtime = None
                continue

            # Detect which kernel block we are in
            for pattern, key in kernel_patterns:
                if line.startswith(pattern):
                    current_kernel_key = key
                    temp_runtime = None
                    break
            if current_kernel_key is None:
                # Not in a known kernel block; skip until we find one
                pass

            # Parse Avg runtime
            if line.startswith("Avg runtime:") and current_kernel_key is not None:
                # Example line: "Avg runtime: 0.00409395 ms"
                try:
                    parts = line.split()
                    # parts = ["Avg", "runtime:", "0.00409395", "ms"]
                    temp_runtime = float(parts[2])
                except (IndexError, ValueError):
                    # If parsing fails, just skip this line
                    temp_runtime = None

            # Parse GFLOPs (we append the entry when we see GFLOPs)
            if line.startswith("GFLOPs:") and current_kernel_key is not None:
                # Example line: "GFLOPs: 512.257"
                try:
                    parts = line.split()
                    # parts = ["GFLOPs:", "512.257"]
                    flops = float(parts[1])
                except (IndexError, ValueError):
                    flops = None

                if temp_runtime is not None and flops is not None:
                    runtimes[current_kernel_key].append(temp_runtime)
                    gflops[current_kernel_key].append(flops)
                    temp_runtime = None  # reset for safety

    # Convert everything to numpy arrays
    bn_values = np.array(bn_list, dtype=np.int32)
    n_values = np.array(n_list, dtype=np.int32)
    for key in kernel_order:
        runtimes[key] = np.asarray(runtimes[key], dtype=np.float32)
        gflops[key] = np.asarray(gflops[key], dtype=np.float32)

    return bn_values, n_values, runtimes, gflops


def plot_results(bn, n_values, runtimes, gflops, out_prefix="streamk_analysis"):
    """
    Create two plots:
      1) GFLOPs vs N
      2) Runtime vs N (ms)

    bn, n_values are 1D arrays.
    runtimes, gflops are dicts: kernel_key -> 1D array over BN.
    """

    # Professional-style plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette(sns.color_palette("colorblind"))

    # Human-readable labels in consistent order
    kernel_labels = {
        "basic_dp": "Basic DP GEMM",
        "streamk_default": "StreamK (default LB)",
        # "streamk_emul_dp": "StreamK emulating DP",
        "basic_splitk": "Basic Split-K (factor 2)",
        # "streamk_emul_splitk": "StreamK emulating Split-K (factor 2)",
    }
    kernel_order = list(kernel_labels.keys())

    # -------- Plot 1: GFLOPs vs N --------
    fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)

    for idx, key in enumerate(kernel_order):
        y = gflops[key]
        x = n_values[: len(y)]  # safety if lengths differ
        marker = "o" if (idx % 2) == 0 else "s"
        linestyle = "-" if (idx % 2) == 0 else "--"

        ax1.plot(
            x,
            y,
            linewidth=2.5,
            linestyle=linestyle,
            label=kernel_labels[key],
            marker=marker,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )

    ax1.set_xlabel("K", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Performance [GFLOP/s]", fontsize=12, fontweight="bold")
    ax1.set_title("GEMM Performance vs Problem Size", fontsize=13, fontweight="bold", pad=10)

    ax1.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
    ax1.set_axisbelow(True)

    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax1.tick_params(axis="both", which="major", labelsize=10, width=1.2)

    legend1 = ax1.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
    )
    legend1.get_frame().set_facecolor("white")
    legend1.get_frame().set_alpha(0.9)

    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_gflops.png", dpi=300, bbox_inches="tight")

    # -------- Plot 2: Runtime vs N --------
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)

    for idx, key in enumerate(kernel_order):
        y = runtimes[key]
        x = n_values[: len(y)]
        marker = "o" if (idx % 2) == 0 else "s"
        linestyle = "-" if (idx % 2) == 0 else "--"

        ax2.plot(
            x,
            y,
            linewidth=2.5,
            linestyle=linestyle,
            label=kernel_labels[key],
            marker=marker,
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )

    ax2.set_xlabel("K", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Avg runtime [ms]", fontsize=12, fontweight="bold")
    ax2.set_title("GEMM Runtime vs Problem Size", fontsize=13, fontweight="bold", pad=10)

    ax2.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.3)
    ax2.set_axisbelow(True)

    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax2.tick_params(axis="both", which="major", labelsize=10, width=1.2)

    legend2 = ax2.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
    )
    legend2.get_frame().set_facecolor("white")
    legend2.get_frame().set_alpha(0.9)

    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_runtime.png", dpi=300, bbox_inches="tight")

    print(f"Saved plots to: {out_prefix}_gflops_K.png and {out_prefix}_runtime_K.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_streamk_results.py <logfile.txt>")
        sys.exit(1)

    logfile = Path(sys.argv[1])
    if not logfile.is_file():
        print(f"Error: file not found: {logfile}")
        sys.exit(1)

    bk, k_values, runtimes, gflops = parse_results(logfile)

    # Only show 3 kernel variants
    desired_keys = ["basic_dp", "streamk_default", "basic_splitk"]

    filtered_runtimes = {k: v for k, v in runtimes.items() if k in desired_keys}
    filtered_gflops   = {k: v for k, v in gflops.items()   if k in desired_keys}

    print(f"Parsed {len(bk)} BK points (K sweep)")
    for key in desired_keys:
        print(
            f"{key:20s}: runtimes shape={filtered_runtimes[key].shape}, "
            f"GFLOPs shape={filtered_gflops[key].shape}"
        )

    out_prefix = logfile.with_suffix("").as_posix()
    plot_results(bk, k_values, filtered_runtimes, filtered_gflops, out_prefix=out_prefix)


if __name__ == "__main__":
    main()
