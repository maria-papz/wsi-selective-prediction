# scripts/build_r2_retained_figure.py
"""
R2 retained-set performance by uncertainty family (ensemble predictor) as a
function of achieved coverage - Figure fig:r2-retained.

Standalone - reads only outputs/selective_prediction_bench/selective_metrics_cal_locked_ensemble.csv.

Same UCL-palette convention as build_r0_figures.py / build_r2_biggest_gain_figure.py.

Input:
    outputs/selective_prediction_bench/selective_metrics_cal_locked_ensemble.csv

Output:
    outputs/r2_figures/retained_metrics_by_family_ensemble.pdf
    outputs/r2_figures/retained_metrics_by_family_ensemble.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/cs/student/project_msc/2025/aibh/mpapageo")
SRC = BASE / "outputs/selective_prediction_bench/selective_metrics_cal_locked_ensemble.csv"
OUT = BASE / "outputs/r2_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

PLOT_METRICS = ["auroc", "sensitivity", "specificity", "ece", "cal_gap", "brier"]
METRIC_TITLE = {
    "auroc": "AUROC", "sensitivity": "Sensitivity", "specificity": "Specificity",
    "ece": "ECE", "cal_gap": "Cal. gap", "brier": "Brier",
}

# UCL brand palette (same as build_r0_figures.py) + a distinct marker shape
# per family -- 6 families need more than colour/linestyle alone to stay
# separable, especially once printed in greyscale or viewed by a colourblind
# reader.
FAMILY_STYLE = {
    "predictive":    dict(color="#361a54", ls="-",  marker="o"),  # dark purple, circle
    "deep_ensemble": dict(color="#993bff", ls="--", marker="s"),  # bright purple, square
    "laplace":       dict(color="#ba82ff", ls=":",  marker="^"),  # mid purple, triangle
    "mc_dropout":    dict(color="#30d6ff", ls="-.", marker="D"),  # heritage blue, diamond
    "epistemic":     dict(color="#AC145A", ls="-",  marker="P"),  # magenta, plus (thick)
    "ood":           dict(color="#2a2a2a", ls="-",  marker="X"),  # dark grey, X
}
FAM_LABEL = {
    "predictive": "Predictive", "deep_ensemble": "Deep ensemble", "laplace": "Laplace",
    "mc_dropout": "MC-dropout", "epistemic": "Epistemic", "ood": "OOD",
}

out = pd.read_csv(SRC)
families = [f for f in FAMILY_STYLE if f in out["family"].unique()]

fig, axes = plt.subplots(3, 2, figsize=(13, 16.5))

for ax, metric in zip(axes.ravel(), PLOT_METRICS):
    base = out[out["metric"] == metric]
    if base.empty or not np.isfinite(base["full"].iloc[0]):
        ax.axis("off")
        continue

    for family in families:
        style = FAMILY_STYLE[family]
        g = base[base["family"] == family].sort_values("achieved_coverage")

        ax.fill_between(g["achieved_coverage"], g["random_lo"], g["random_hi"],
                         alpha=0.06, color=style["color"], zorder=1)
        ax.plot(g["achieved_coverage"], g["random_mean"],
                 linestyle=":", linewidth=0.9, alpha=0.6, color=style["color"], zorder=2)
        ax.plot(g["achieved_coverage"], g["retained"],
                 marker=style["marker"], markersize=8.5, linewidth=1.6,
                 linestyle=style["ls"], color=style["color"],
                 label=FAM_LABEL[family], zorder=3)

    ax.axhline(base["full"].iloc[0], linestyle="--", linewidth=1.3,
               color="black", label="no abstention")

    ax.set(xlabel="achieved evaluation coverage", title=METRIC_TITLE[metric])
    ax.grid(alpha=0.3)

axes.ravel()[0].legend(fontsize=10.5, loc="best", framealpha=0.9)

fig.suptitle(
    "Retained-set metrics using calibration-locked cutoffs — ensemble predictor\n"
    "Dotted lines and shaded bands show each family's random-deferral reference "
    "distribution at its own matched achieved coverage",
    fontsize=14,
)

plt.tight_layout(rect=(0, 0, 1, 0.94))

for ext in ("pdf", "png"):
    out_path = OUT / f"retained_metrics_by_family_ensemble.{ext}"
    kwargs = {"dpi": 300} if ext == "png" else {}
    plt.savefig(out_path, **kwargs)
    print(f"Saved {out_path}")
