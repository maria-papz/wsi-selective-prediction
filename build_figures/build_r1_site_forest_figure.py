# scripts/build_r1_site_forest_figure.py
"""
R1 per-site forest plot -- ensemble AUROC by TCGA tissue source site, with
bootstrap 95% CIs, TJU (Thomas Jefferson University) called out as the
outlier site.

Standalone -- reads only outputs/selective_prediction_bench/master_patient_table.csv
(patient-level: site, label, ensemble prob_mutant) and applies the same
CALIBRATION_SITES exclusion as build_r0_figures.py, so it doesn't need the
notebook re-run to regenerate the figure. Bootstrap CIs are computed here,
not read from a saved CSV.

Input:
    outputs/selective_prediction_bench/master_patient_table.csv

Output:
    outputs/r1_figures/baseline_site_forest_plot.png
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path("/cs/student/project_msc/2025/aibh/mpapageo")
SRC = BASE / "outputs/selective_prediction_bench/master_patient_table.csv"
FIG_DIR = BASE / "outputs/r1_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
})

# UCL brand palette (same as build_r0_figures.py).
DARK_PURPLE = "#361a54"
BRIGHT_PURPLE = "#993bff"
GREY = "#888888"

TJU = "Thomas Jefferson University"
MIN_SITE_N = 10   # site-inclusion floor -- matches "14 sites (n>=10)" in the Results prose
MIN_CLASS_N = 3    # per-site AUROC needs >=3 of each class, else it's excluded (still counted
                    # in the 14-site/631-patient total, just not plottable)
N_BOOT = 2000
SEED = 0

# Must match build_r0_figures.py's CALIBRATION_SITES exactly -- "Fondazione-Besta"
# and the full Milan name are the same institution under two spellings in
CALIBRATION_SITES = {
    "University of Florida",
    "Milan - Italy, Fondazione IRCCS Instituto Neuroligico C. Besta",
    "Fondazione-Besta",
    "Mayo Clinic - Rochester",
}

raw = pd.read_csv(SRC)
evalset = raw[~raw["tissue_source_site.name"].isin(CALIBRATION_SITES)].copy()

site_n = evalset["tissue_source_site.name"].value_counts()
big_sites = site_n[site_n >= MIN_SITE_N].index
big = evalset[evalset["tissue_source_site.name"].isin(big_sites)]

print(f"Eval cohort: {len(evalset)} patients | "
      f"{len(big_sites)} sites with n>={MIN_SITE_N}, {len(big)} patients "
      f"({sorted(big_sites)})")


def site_auroc_ci(y, p, n_boot=N_BOOT, seed=SEED):
    if min((y == 0).sum(), (y == 1).sum()) < MIN_CLASS_N:
        return np.nan, np.nan, np.nan
    obs = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if min((yb == 0).sum(), (yb == 1).sum()) < MIN_CLASS_N:
            continue
        boots.append(roc_auc_score(yb, p[idx]))
    if len(boots) < 100:
        return obs, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, lo, hi


rows, no_auroc = [], []
for site, g in big.groupby("tissue_source_site.name"):
    y = g["label"].to_numpy(int)
    p = g["prob_mutant"].to_numpy(float)
    obs, lo, hi = site_auroc_ci(y, p)
    if np.isnan(obs):
        no_auroc.append(site)
        continue
    rows.append({"site": site, "n": len(g), "prev": y.mean(),
                 "auroc": obs, "lo": lo, "hi": hi,
                 "degen": np.isfinite(hi) and (hi - lo) < 0.01})

df = pd.DataFrame(rows).sort_values("auroc").reset_index(drop=True)
prev_all = evalset[evalset["tissue_source_site.name"].isin(big_sites)] \
    .groupby("tissue_source_site.name")["label"].mean()

print(f"Excluded from AUROC (< {MIN_CLASS_N} of one class): {no_auroc}")
print(f"IDH-mut prevalence across the {len(big_sites)} n>={MIN_SITE_N} sites: "
      f"{prev_all.min():.2f}-{prev_all.max():.2f}")

# ---- manual multi-column forest plot -----------------------------------
n_rows = len(df)
fig, ax = plt.subplots(figsize=(9, max(4.2, 0.5 * n_rows)))
y_pos = np.arange(n_rows)

xerr = np.vstack([
    (df["auroc"] - df["lo"]).clip(lower=0),
    (df["hi"] - df["auroc"]).clip(lower=0),
])
ax.errorbar(df["auroc"], y_pos, xerr=xerr, fmt="none",
            ecolor=DARK_PURPLE, alpha=0.5, elinewidth=1, capsize=3, zorder=2)
colors = [BRIGHT_PURPLE if s == TJU else DARK_PURPLE for s in df["site"]]
ax.scatter(df["auroc"], y_pos, c=colors, s=40, zorder=3)

for i, degen in enumerate(df["degen"]):
    if degen:
        ax.annotate("degenerate CI (small n)", (df["auroc"].iloc[i], y_pos[i]),
                    xytext=(6, 6), textcoords="offset points", fontsize=6.5,
                    color=GREY, style="italic")

ax.axvline(0.5, color=GREY, lw=0.8, ls=":", zorder=1)
ax.text(0.505, 0.94, "Chance", transform=ax.get_xaxis_transform(),
        ha="left", va="top", fontsize=7, color=GREY, style="italic")

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{s}  (n={n}, prevalence={p:.2f})"
                     for s, n, p in zip(df["site"], df["n"], df["prev"])])
ax.set_xlabel("AUROC (95% bootstrap CI)")
ax.set_title(f"Per-site discrimination — ensemble predictor (n ≥ {MIN_SITE_N})")
ax.set_xlim(0, 1.02)  
ax.grid(axis="x", alpha=0.25, color=GREY)

plt.tight_layout()

for ext in ("png", "pdf"):
    out_path = FIG_DIR / f"baseline_site_forest_plot.{ext}"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {out_path}")
