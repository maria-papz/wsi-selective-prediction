# scripts/build_r3a_uni2_figure.py
"""
R3a UNI2-by-family figure -- replaces tab:r3a-uni2 (18-row table: 6 families
x 3 coverages x {AUROC delta, accuracy delta}). A chart makes the section's
actual point -- AUROC and accuracy move in opposite directions as coverage
tightens, and neither clears its MDE -- visible directly, instead of making
the reader scan 18 rows of near-identical deltas to notice the pattern.

Grouped bar chart: one cluster of bars per target coverage, one bar per
uncertainty family within each cluster, two panels (AUROC delta / accuracy
delta). Bars that don't clear their minimum detectable effect (MDE) are
shown faded and hatched, not just a plain bar, so a "no effect" cell doesn't
read the same as a real one at a glance.

Standalone - reads only outputs/ipd_brain_baseline_bench/uncertainty_deferral_locked_vs_recalibrated.csv,
restricted to predictor='uni2' .


Output:
    outputs/r3_figures/r3a_uni2_by_family.pdf
    outputs/r3_figures/r3a_uni2_by_family.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/cs/student/project_msc/2025/aibh/mpapageo")
SRC = BASE / "outputs/ipd_brain_baseline_bench/uncertainty_deferral_locked_vs_recalibrated.csv"
OUT = BASE / "outputs/r3_figures"
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


FAMILY_COLOR = {
    "predictive":    "#361a54",
    "deep_ensemble": "#993bff",
    "laplace":       "#ba82ff",
    "mc_dropout":    "#30d6ff",
    "epistemic":     "#AC145A",
    "ood":           "#2a2a2a",
}
FAM_LABEL = {
    "predictive": "Predictive", "deep_ensemble": "Deep ensemble", "laplace": "Laplace",
    "mc_dropout": "MC-dropout", "epistemic": "Epistemic", "ood": "OOD",
}
FAM_ORDER = ["predictive", "deep_ensemble", "laplace", "mc_dropout", "epistemic", "ood"]
COVERAGES = [0.7, 0.8, 0.9]
COV_LABEL = {0.7: "70%", 0.8: "80%", 0.9: "90%"}

df = pd.read_csv(SRC)
df = df[df["predictor"] == "uni2"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

panels = [
    ("primary_d_full_auroc", "primary_retained_auroc_mde", axes[0], "$\\Delta$ AUROC vs. full cohort"),
    ("primary_d_full_accuracy", "primary_retained_accuracy_mde", axes[1], "$\\Delta$ accuracy vs. full cohort"),
]

n_fam = len(FAM_ORDER)
bar_w = 0.8 / n_fam
group_centers = np.arange(len(COVERAGES))

for delta_col, mde_col, ax, title in panels:
    ax.axhline(0, color="#333333", lw=1.6, zorder=4)
    for fi, family in enumerate(FAM_ORDER):
        color = FAMILY_COLOR[family]
        g = df[df["family"] == family].set_index("target_coverage")
        xs = group_centers - 0.4 + bar_w * (fi + 0.5)
        heights = [g.loc[cov, delta_col] for cov in COVERAGES]
        clears = [abs(g.loc[cov, delta_col]) >= g.loc[cov, mde_col] for cov in COVERAGES]

        for x, h, sig in zip(xs, heights, clears):
            if sig:
                ax.bar(x, h, width=bar_w * 0.92, color=color, edgecolor="none", zorder=3)
            else:
                # Not "faint" -- still a clearly readable bar at its real
                # height, just marked non-significant by hatching + a
                # lighter (not washed-out) fill.
                ax.bar(x, h, width=bar_w * 0.92, color=color, alpha=0.6,
                       edgecolor=color, linewidth=0.8, hatch="///", zorder=3)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([COV_LABEL[c] for c in COVERAGES])
    ax.set_xlabel("target coverage")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("$\\Delta$ (retained $-$ full cohort)")


handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in FAM_ORDER]
fig.legend(handles, [FAM_LABEL[f] for f in FAM_ORDER], loc="upper center",
           ncol=len(FAM_ORDER), bbox_to_anchor=(0.5, 1.02), framealpha=0.9, fontsize=9)

plt.tight_layout(rect=(0, 0, 1, 0.90))

for ext in ("pdf", "png"):
    out_path = OUT / f"r3a_uni2_by_family.{ext}"
    kwargs = {"dpi": 300} if ext == "png" else {}
    plt.savefig(out_path, **kwargs)
    print(f"Saved {out_path}")
