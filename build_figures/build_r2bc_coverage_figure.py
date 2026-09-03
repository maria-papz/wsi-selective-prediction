# scripts/build_r2bc_coverage_figure.py
"""
R2b/R2c conformal coverage figure, TCGA institutional shift (panel A) and
IPD Brain external shift (panel B), side by side with identical axes,
markers, ordering, legend terminology, and y-limits, so the two shifts are
directly visually comparable. 

Standalone -- reads only:
    outputs/shift_weighted_conformal_leakage_free/coverage_comparison.csv (TCGA)
    outputs/shift_weighted_conformal_ipd_brain_extension/coverage_results.csv (IPD Brain)
restricted to the three schemes each table covers (plain, UNI2-weighted,
H-optimus-weighted).


Output:
    outputs/r2b_figures/r2bc_coverage.pdf
    outputs/r2b_figures/r2bc_coverage.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/cs/student/project_msc/2025/aibh/mpapageo")
OUT = BASE / "outputs/r2b_figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

SCHEME_STYLE = {
    "plain":            dict(color="#2a2a2a", marker="o", label="Plain"),
    "uni2":             dict(color="#993bff", marker="s", label="UNI2-weighted"),
    "hoptimus":         dict(color="#30d6ff", marker="^", label="H-optimus-weighted"),
}
SCHEME_ORDER = ["plain", "uni2", "hoptimus"]

# --- TCGA (panel A) ---
tcga = pd.read_csv(BASE / "outputs/shift_weighted_conformal_leakage_free/coverage_comparison.csv")
tcga_map = {"plain": "plain", "weighted_uni2only": "uni2", "weighted_hoptimusonly": "hoptimus"}
tcga = tcga[tcga["method"].isin(tcga_map)].copy()
tcga["scheme"] = tcga["method"].map(tcga_map)

# --- IPD Brain (panel B) ---
ipd = pd.read_csv(BASE / "outputs/shift_weighted_conformal_ipd_brain_extension/coverage_results.csv")
ipd_map = {"plain": "plain", "weighted_uni2": "uni2", "weighted_hoptimus": "hoptimus"}
ipd = ipd[ipd["method"].isin(ipd_map)].copy()
ipd["scheme"] = ipd["method"].map(ipd_map)

pooled = pd.concat([tcga, ipd], ignore_index=True)
y_range = pooled["ci_hi"].max() - pooled["ci_lo"].min()
Y_LO = pooled["ci_lo"].min() - 0.06 * y_range
Y_HI = pooled["ci_hi"].max() + 0.12 * y_range  # extra headroom: significance stars sit above the marker
x_range = pooled["target_coverage"].max() - pooled["target_coverage"].min()
X_LO = pooled["target_coverage"].min() - 0.28 * x_range
X_HI = pooled["target_coverage"].max() + 0.28 * x_range

def sig_stars(p):
    if p <= 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""



targets = np.sort(pooled["target_coverage"].unique())
min_spacing = np.diff(targets).min()
dodge_step = 0.06 * min_spacing  # 6% of the gap between adjacent targets
DODGE = {"plain": -dodge_step, "uni2": 0.0, "hoptimus": dodge_step}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, data, title in [
    (axes[0], tcga, "A -- TCGA institutional shift ($n=337$)"),
    (axes[1], ipd, "B -- IPD Brain external shift ($n=175$ scored)"),
]:
    diag_lo, diag_hi = min(X_LO, Y_LO), max(X_HI, Y_HI)
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], ls="--", lw=1.2, color="#888888", zorder=1,
            label="nominal target (achieved = target)")
    for scheme in SCHEME_ORDER:
        style = SCHEME_STYLE[scheme]
        g = data[data["scheme"] == scheme].sort_values("target_coverage")
        x = g["target_coverage"] + DODGE[scheme]
        yerr = np.vstack([g["coverage"] - g["ci_lo"], g["ci_hi"] - g["coverage"]])
        _, caplines, barlinecols = ax.errorbar(
            x, g["coverage"], yerr=yerr, fmt=style["marker"],
            markersize=11, markeredgewidth=0, color=style["color"], capsize=4,
            elinewidth=1.3, linewidth=1.8, linestyle="-", label=style["label"], zorder=3)

        for cap in caplines:
            cap.set_alpha(0.45)
        for barlinecol in barlinecols:
            barlinecol.set_alpha(0.45)


        if scheme == "plain":
            continue
        for xi, (_, row) in zip(x, g.iterrows()):
            stars = sig_stars(row["p"])
            if not stars:
                continue
            ax.annotate(stars, (xi, row["coverage"]),
                        textcoords="offset points", xytext=(0, 13), ha="center",
                        va="bottom", fontsize=10, color=style["color"], fontweight="bold")

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks([0.7, 0.8, 0.9])
    ax.set_xticklabels(["70%", "80%", "90%"])
    ax.set_xlabel("target coverage")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("achieved coverage (95% CI)")
axes[1].legend(loc="lower right", framealpha=0.9)

plt.tight_layout()
for ext in ("pdf", "png"):
    out_path = OUT / f"r2bc_coverage.{ext}"
    kwargs = {"dpi": 300} if ext == "png" else {}
    plt.savefig(out_path, **kwargs)
    print(f"Saved {out_path}")
