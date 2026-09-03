# scripts/build_r2_family_heatmap_90.py
"""
R2 predictor x abstention-family heatmap, 90% coverage only -- a legible
replacement for both the old single-winner heatmap (apples-and-oranges,
per supervisor) and the full 18-panel 6-metric x 3-coverage grid this
script's sibling (build_r2_family_heatmap_grid.py) produced, which turned
out too dense to actually read. Scoped to 90% coverage to match
tab:r2-metric-breakdown (also 90%-only) -- this figure supplies the
magnitude the table's "n/5 significant" counts don't show, at the one
coverage already anchoring that table, rather than trying to cover all
three coverages in one image.

Standalone -- reads only outputs/selective_prediction_bench/selective_metrics_cal_locked_all_fdr.csv.

Same UCL-palette convention as build_r0_figures.py / build_r2_family_heatmap_grid.py.

Input:
    outputs/selective_prediction_bench/selective_metrics_cal_locked_all_fdr.csv

Output:
    outputs/r2_figures/r2_family_heatmap_90pct.pdf
    outputs/r2_figures/r2_family_heatmap_90pct.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BASE = Path("/cs/student/project_msc/2025/aibh/mpapageo")
SRC = BASE / "outputs/selective_prediction_bench/selective_metrics_cal_locked_all_fdr.csv"
OUT = BASE / "outputs/r2_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     13,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})


METRICS = ["sensitivity", "specificity", "auroc", "bss"]
METRIC_LABEL = {
    "sensitivity": "Sensitivity", "specificity": "Specificity", "auroc": "AUROC",
    "bss": "BSS",
}
LOWER_BETTER = {"ece", "cal_gap"}
TARGET_COVERAGE = 0.9

PREDICTORS = ["uni2", "conch", "hoptimus", "uni2+hoptimus", "ensemble"]
PRED_LABEL = {
    "uni2": "UNI2", "conch": "CONCH", "hoptimus": "H-optimus",
    "uni2+hoptimus": "UNI2+H-opt.", "ensemble": "Ensemble",
}
FAMILIES = ["predictive", "deep_ensemble", "laplace", "mc_dropout", "epistemic", "ood"]
FAM_LABEL = {
    "predictive": "Predictive", "deep_ensemble": "Deep ens.", "laplace": "Laplace",
    "mc_dropout": "MC-drop.", "epistemic": "Epistemic", "ood": "OOD",
}

UCL_SEQ = LinearSegmentedColormap.from_list("ucl_seq", ["#ffffff", "#AC145A", "#500778", "#2C0442"])

df = pd.read_csv(SRC)
df90 = df[df["target_coverage"] == TARGET_COVERAGE]

fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))

for ax, metric in zip(axes.ravel(), METRICS):
    sign = -1 if metric in LOWER_BETTER else 1
    g_raw = df90[df90["metric"] == metric]

    grid = np.full((len(PREDICTORS), len(FAMILIES)), np.nan)
    for pi, pred in enumerate(PREDICTORS):
        for fi, fam in enumerate(FAMILIES):
            r = g_raw[(g_raw["predictor"] == pred) & (g_raw["family"] == fam)]
            if len(r) and np.isfinite(r["delta_vs_full"].iloc[0]):
                grid[pi, fi] = sign * r["delta_vs_full"].iloc[0]

    vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, cmap=UCL_SEQ, vmin=0, vmax=vmax, aspect="auto")

    for pi, pred in enumerate(PREDICTORS):
        for fi, fam in enumerate(FAMILIES):
            r = g_raw[(g_raw["predictor"] == pred) & (g_raw["family"] == fam)]
            if not len(r) or not np.isfinite(r["delta_vs_full"].iloc[0]):
                ax.text(fi, pi, "n/a", ha="center", va="center", fontsize=9, color="#999")
                continue
            raw_val = r["delta_vs_full"].iloc[0]
            sig = bool(r["sig_fdr"].iloc[0])
            disp = grid[pi, fi]
            txt_color = "white" if (np.isfinite(disp) and disp > 0.55 * vmax) else "black"
            weight = "bold" if sig else "normal"
            ax.text(fi, pi, f"{raw_val:+.3f}", ha="center", va="center",
                     fontsize=11, color=txt_color, fontweight=weight)
            if not sig:
                ax.add_patch(plt.Rectangle((fi - 0.5, pi - 0.5), 1, 1, fill=False,
                                            hatch="////", edgecolor="#bbbbbb", linewidth=0))

    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([FAM_LABEL[f] for f in FAMILIES], rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(PREDICTORS)))
    ax.set_yticklabels([PRED_LABEL[p] for p in PREDICTORS], fontsize=10)
    ax.set_title(METRIC_LABEL[metric], fontsize=13)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(FAMILIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PREDICTORS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout(rect=(0, 0, 1, 0.93))
fig.suptitle(
    "Retained-set improvement over no-abstention baseline at 90% target coverage,\n"
    "every predictor x abstention family, TCGA "
    "(bold = significant vs. random deferral at BH-FDR $q<0.05$; hatched = not significant)",
    fontsize=13, y=0.995,
)

for ext in ("pdf", "png"):
    out_path = OUT / f"r2_family_heatmap_90pct_core4.{ext}"
    kwargs = {"dpi": 300} if ext == "png" else {}
    plt.savefig(out_path, **kwargs)
    print(f"Saved {out_path}")
