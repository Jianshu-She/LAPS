"""Plot benchmark results from raw JSON files.

Usage:
    python plot_results.py <results_dir>

Reads raw/*.json files and generates comparison plots for all metrics.
Saves plots to <results_dir>/plots/.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_results(raw_dir, settings, ccs):
    """Load all JSON result files into a nested dict: {setting: {cc: data}}."""
    results = {}
    for s in settings:
        results[s] = {}
        for cc in ccs:
            path = os.path.join(raw_dir, f"{s}_cc{cc}.json")
            try:
                with open(path) as f:
                    results[s][cc] = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results[s][cc] = None
    return results


def get_values(results, settings, ccs, key):
    """Extract a metric across all cc levels for each setting."""
    out = {}
    for s in settings:
        vals = []
        for cc in ccs:
            d = results[s].get(cc)
            vals.append(d[key] if d and key in d else None)
        out[s] = vals
    return out


SETTING_LABELS = {
    "vanilla_sglang": "Vanilla SGLang",
    "disaggregation": "Disaggregation",
    "laps": "LAPS",
}

SETTING_COLORS = {
    "vanilla_sglang": "#4C72B0",
    "disaggregation": "#DD8452",
    "laps": "#55A868",
}

SETTING_MARKERS = {
    "vanilla_sglang": "o",
    "disaggregation": "s",
    "laps": "D",
}

METRICS = [
    ("request_throughput_req_s", "Request Throughput (req/s)"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("p90_ttft_ms", "P90 TTFT (ms)"),
]


def plot_metric(ax, results, settings, ccs, key, ylabel):
    """Plot a single metric on the given axes."""
    x = np.arange(len(ccs))
    for s in settings:
        vals = []
        valid_x = []
        for i, cc in enumerate(ccs):
            d = results[s].get(cc)
            if d and key in d:
                vals.append(d[key])
                valid_x.append(x[i])
        if vals:
            ax.plot(
                valid_x, vals,
                marker=SETTING_MARKERS.get(s, "o"),
                color=SETTING_COLORS.get(s, None),
                label=SETTING_LABELS.get(s, s),
                linewidth=2,
                markersize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in ccs])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("results_dir", help="Path to results directory (containing raw/)")
    parser.add_argument("--settings", default="vanilla_sglang,disaggregation,laps",
                        help="Comma-separated list of settings")
    parser.add_argument("--cc", default="1,2,4,8,16,32,64",
                        help="Comma-separated concurrency levels")
    args = parser.parse_args()

    results_dir = args.results_dir
    raw_dir = os.path.join(results_dir, "raw")
    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    settings = args.settings.split(",")
    ccs = [int(c) for c in args.cc.split(",")]

    if not os.path.isdir(raw_dir):
        print(f"ERROR: raw directory not found: {raw_dir}")
        sys.exit(1)

    results = load_results(raw_dir, settings, ccs)

    # Check if we have any data
    has_data = any(results[s][cc] is not None for s in settings for cc in ccs)
    if not has_data:
        print("ERROR: No result files found in raw/")
        sys.exit(1)

    # --- Individual metric plots ---
    for key, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_metric(ax, results, settings, ccs, key, ylabel)
        ax.set_title(ylabel)
        fig.tight_layout()
        fname = os.path.join(plot_dir, f"{key}.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


    print(f"\nAll plots saved to: {plot_dir}/")


if __name__ == "__main__":
    main()
