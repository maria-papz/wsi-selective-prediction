# scripts/build_r2_nnd_figure.py
"""
R2 number-needed-to-defer (NND) and inference-cost figure -- converts
tab:r2-cost into a figure

Standalone - reads only outputs/selective_prediction_bench/number_needed_to_defer.csv
and outputs/selective_prediction_bench/inference_cost_by_family.csv.

Left panel: NND vs. coverage target, UNI2, one line per family (lower is
better - fewer patients need deferring to prevent one error).
Right panel: inference cost relative to a single forward pass, log scale
(spans 1x to ~333x, so a linear axis would flatten everything but MC-dropout
to the baseline). Epistemic has no separate cost (piggybacks on other
families' forward passes already computed) - shown as a labelled gap,
not a bar at 0, so it isn't misread as free.

Input:
    outputs/selective_prediction_bench/number_needed_to_defer.csv
    outputs/selective_prediction_bench/inference_cost_by_family.csv

Output:
    outputs/r2_figures/r2_nnd_cost.pdf
    outputs/r2_figures/r2_nnd_cost.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/cs/student/project_msc/2025/aibh/mpapageo")
BENCH = BASE / "outputs/selective_prediction_bench"
OUT = BASE / "outputs/r2_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

FAMILY_STYLE = {
    "predictive":    dict(color="#361a54", marker="o"),
    "deep_ensemble": dict(color="#993bff", marker="s"),
    "laplace":       dict(color="#ba82ff", marker="^"),
    "mc_dropout":    dict(color="#30d6ff", marker="D"),
    "epistemic":     dict(color="#AC145A", marker="P"),
    "ood":           dict(color="#2a2a2a", marker="X"),
}
FAM_LABEL = {
    "predictive": "Predictive", "deep_ensemble": "Deep ensemble", "laplace": "Laplace",
    "mc_dropout": "MC-dropout", "epistemic": "Epistemic", "ood": "OOD",
}
FAM_ORDER = ["predictive", "deep_ensemble", "laplace", "mc_dropout", "epistemic", "ood"]

nnd = pd.read_csv(BENCH / "number_needed_to_defer.csv")
nnd_uni2 = nnd[nnd["predictor"] == "uni2"]

cost = pd.read_csv(BENCH / "inference_cost_by_family.csv")
# map the cost file's descriptive family names onto the same short keys
# used everywhere else in this section
COST_KEY = {
    "predictive / cross-fold-cross-encoder": "predictive",
    "density-based OOD": "ood",
    "MC-dropout": "mc_dropout",
    "deep ensemble": "deep_ensemble",
    "last-layer Laplace": "laplace",
}
cost = cost[cost["family"].isin(COST_KEY)].copy()
cost["family_key"] = cost["family"].map(COST_KEY)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# --- left: NND vs coverage, UNI2 ---
for family in FAM_ORDER:
    g = nnd_uni2[nnd_uni2["family"] == family].sort_values("target_coverage")
    style = FAMILY_STYLE[family]
    ax1.plot(g["target_coverage"] * 100, g["nnd"], marker=style["marker"],
              markersize=10, linewidth=2, color=style["color"], label=FAM_LABEL[family])

ax1.set_xlabel("target coverage (%)")
ax1.set_ylabel("number needed to defer (UNI2)")
ax1.set_title("Fewer is better")
ax1.set_xticks([70, 80, 90])
ax1.grid(alpha=0.3)
ax1.legend(loc="upper left", framealpha=0.9)

# --- right: inference cost, log scale ---
plot_fams = [f for f in FAM_ORDER if f != "epistemic"]
xs = np.arange(len(plot_fams))
vals = [cost.loc[cost["family_key"] == f, "relative_to_single_pass"].iloc[0] for f in plot_fams]
colors = [FAMILY_STYLE[f]["color"] for f in plot_fams]

bars = ax2.bar(xs, vals, color=colors, width=0.6)
ax2.set_yscale("log")
ax2.set_xticks(xs)
ax2.set_xticklabels([FAM_LABEL[f] for f in plot_fams], rotation=30, ha="right")
ax2.set_ylabel("inference cost (x single forward pass, log scale)")
ax2.set_title("Fewer is cheaper")
ax2.grid(axis="y", alpha=0.3, which="both")

for x, v in zip(xs, vals):
    ax2.annotate(f"{v:.2f}x" if v < 10 else f"{v:.0f}x", (x, v),
                 xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)

# Epistemic: no separate forward pass -- annotate the gap instead of a bar
ax2.text(0.5, 0.97, "Epistemic: no separate cost\n(reuses another family's passes)",
          transform=ax2.get_xaxis_transform(), fontsize=8.5,
          color=FAMILY_STYLE["epistemic"]["color"], ha="center", va="top", style="italic")

fig.suptitle("Cost of abstention: number needed to defer vs. inference overhead, TCGA", fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.94))

for ext in ("pdf", "png"):
    out_path = OUT / f"r2_nnd_cost.{ext}"
    kwargs = {"dpi": 300} if ext == "png" else {}
    plt.savefig(out_path, **kwargs)
    print(f"Saved {out_path}")
