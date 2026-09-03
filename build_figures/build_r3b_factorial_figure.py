# scripts/build_r3b_factorial_figure.py
"""
R3b factorial-analysis figure: gains/losses breakdowns (by metric, by
scenario, by family) recomputed directly from the significance-testing
output. 

Source: uncertainty_overall_threshold_combo.csv, filtered to
combo in {"external", "fully_local"} (TCGA-locked vs. IPD-recalibrated),
giving exactly the 2262 = 29 predictor-family pairs x 3 coverages x
2 scenarios x 13 metrics combinations the table describes. "Gain"/"loss"
direction follows d_random's sign, flipped for the three lower-is-better
metrics (brier, ece, cal_gap).

Seven metrics (auroc, auprc, npv, brier, ece, cal_gap, bss) show 0 gains
and 0 losses throughout and are omitted from panel A's bars 

Input:
    outputs/ipd_brain_baseline_bench/uncertainty_overall_threshold_combo.csv

Output:
    outputs/r3_figures/r3b_factorial_gains_losses.pdf
    outputs/r3_figures/r3b_factorial_gains_losses.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs/r3_figures"
OUT.mkdir(parents=True, exist_ok=True)

GAIN_COLOR = "#5b3d99"   # UCL bright purple
LOSS_COLOR = "#AC145A"   # UCL mid purple/magenta
LOWER_BETTER = {"brier", "ece", "cal_gap"}

combo = pd.read_csv(BASE / "outputs/ipd_brain_baseline_bench/uncertainty_overall_threshold_combo.csv")
combo = combo[combo["combo"].isin(["external", "fully_local"])].copy()
combo["is_gain"] = np.where(combo["metric"].isin(LOWER_BETTER),
                             combo["d_random"] < 0, combo["d_random"] > 0)
sig = combo[combo["sig_fdr"]]

HEADLINE_TOTAL, HEADLINE_GAINS, HEADLINE_LOSSES = len(combo), int(sig["is_gain"].sum()), int((~sig["is_gain"]).sum())
HEADLINE_SIG = len(sig)

METRIC_ORDER = ["specificity", "ppv", "balanced_accuracy", "accuracy", "f1", "sensitivity"]
METRIC_LABEL = {"specificity": "specificity", "ppv": "ppv", "balanced_accuracy": "balanced accuracy",
                "accuracy": "accuracy", "f1": "f1", "sensitivity": "sensitivity"}
METRIC_DATA = {}
for m in METRIC_ORDER:
    g = sig[(sig.metric == m) & sig.is_gain]
    l = sig[(sig.metric == m) & ~sig.is_gain]
    METRIC_DATA[METRIC_LABEL[m]] = (len(g), len(l))

SCENARIO_LABEL = {"external": "TCGA-locked", "fully_local": "IPD-recalibrated"}
SCENARIO_DATA = {}
for key, label in SCENARIO_LABEL.items():
    g = sig[(sig.combo == key) & sig.is_gain]
    l = sig[(sig.combo == key) & ~sig.is_gain]
    n = len(combo[combo.combo == key])
    SCENARIO_DATA[label] = (len(g), len(l), n)

FAMILY_ORDER = ["predictive", "laplace", "mc_dropout", "deep_ensemble", "ood", "epistemic"]
FAMILY_LABEL = {"predictive": "predictive", "laplace": "Laplace", "mc_dropout": "MC-dropout",
                 "deep_ensemble": "deep ensemble", "ood": "OOD", "epistemic": "epistemic"}
FAMILY_DATA = {}
for key in FAMILY_ORDER:
    g = sig[(sig.family == key) & sig.is_gain]
    l = sig[(sig.family == key) & ~sig.is_gain]
    n = len(combo[combo.family == key])
    FAMILY_DATA[FAMILY_LABEL[key]] = (len(g), len(l), n)


def diverging_panel(ax, data, title, show_rate=False):
    labels = list(data.keys())
    y = np.arange(len(labels))
    gains = np.array([v[0] for v in data.values()])
    losses = np.array([v[1] for v in data.values()])
    ax.barh(y, gains, color=GAIN_COLOR, label="gains")
    ax.barh(y, -losses, color=LOSS_COLOR, label="losses")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    for yi, g, l in zip(y, gains, losses):
        if g > 0:
            ax.text(g + 1, yi, str(g), va="center", ha="left", fontsize=8.5, color=GAIN_COLOR)
        if l > 0:
            ax.text(-l - 1, yi, str(l), va="center", ha="right", fontsize=8.5, color=LOSS_COLOR)
    xmax = max(gains.max(), losses.max()) * 1.35 + 5
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("$\\leftarrow$ losses      gains $\\rightarrow$", fontsize=9)
    if show_rate:
        for yi, (g, l, n) in zip(y, data.values()):
            ax.text(xmax * 0.98, yi, f"{100*g/n:.1f}%", va="center", ha="right",
                     fontsize=8, color="grey", style="italic")


fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 0.6, 1]})

diverging_panel(axes[0], METRIC_DATA, "By metric")
diverging_panel(axes[1], SCENARIO_DATA, "By scenario")
diverging_panel(axes[2], FAMILY_DATA, "By family (with gain rate)", show_rate=True)

fig.suptitle(f"Gains and losses vs. random-deferral null across {HEADLINE_TOTAL} combinations\n"
             f"{HEADLINE_SIG} significant overall ({HEADLINE_GAINS} gains, {HEADLINE_LOSSES} losses)", fontsize=11)
fig.text(0.5, 0.005,
         "Seven other metrics (AUROC, AUPRC, NPV, Brier, ECE, calibration gap, BSS) show 0 gains and 0 losses throughout.",
         ha="center", fontsize=8, style="italic", color="grey")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT / "r3b_factorial_gains_losses.pdf")
plt.savefig(OUT / "r3b_factorial_gains_losses.png", dpi=150)
print("Saved to", OUT / "r3b_factorial_gains_losses.pdf")
