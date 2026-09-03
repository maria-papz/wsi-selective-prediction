"""
R6 pipeline figure: total-risk two-stage pipeline accuracy and automation
rate vs. per-stage target coverage, TCGA (codel primary) vs. IPD Brain
(codel secondary, ATRX-confident tier). Standalone -- reads only the two
saved two_stage_total_risk_pipeline.csv files
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("/cs/student/project_msc/2025/aibh/mpapageo")
FIG_DIR = BASE / "outputs/r6_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

UCL_STYLE = {
    "TCGA":      dict(color="#361a54", ls="-"),   # dark purple, solid
    "IPD Brain": dict(color="#30d6ff", ls="--"),  # heritage blue, dashed
}

# --- TCGA: codel primary two-stage pipeline ---
tcga = pd.read_csv(BASE / "outputs/selective_prediction_bench_codel/two_stage_total_risk_pipeline.csv")
tcga = tcga.sort_values("target_coverage")

# --- IPD Brain: codel secondary, ATRX-confident tier (the headline tier) ---
ipd = pd.read_csv(BASE / "outputs/codel_ipd_brain_extension/two_stage_total_risk_pipeline.csv")
ipd = ipd[ipd.tier == "ATRX-confident (headline)"].sort_values("target_coverage")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
ax.plot(tcga["target_coverage"], tcga["pipeline_accuracy"], marker="o",
        label="TCGA", **UCL_STYLE["TCGA"])
ax.fill_between(tcga["target_coverage"], tcga["pipeline_accuracy_lo"], tcga["pipeline_accuracy_hi"],
                 color=UCL_STYLE["TCGA"]["color"], alpha=0.15, lw=0)
ax.axhline(tcga["full_set_accuracy"].iloc[0], color=UCL_STYLE["TCGA"]["color"], lw=0.8, alpha=0.4,
           label="TCGA oracle baseline")
ax.plot(ipd["target_coverage"], ipd["pipeline_accuracy"], marker="s",
        label="IPD Brain", **UCL_STYLE["IPD Brain"])
ax.fill_between(ipd["target_coverage"], ipd["pipeline_accuracy_lo"], ipd["pipeline_accuracy_hi"],
                 color=UCL_STYLE["IPD Brain"]["color"], alpha=0.15, lw=0)
ax.axhline(ipd["full_set_accuracy"].iloc[0], color=UCL_STYLE["IPD Brain"]["color"], lw=0.8, alpha=0.4,
           label="IPD Brain oracle baseline")
ax.set(xlabel="per-stage target coverage", ylabel="pipeline accuracy",
       title="Two-stage pipeline accuracy vs. coverage")
ax.invert_xaxis()
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(tcga["target_coverage"], tcga["automation_rate"], marker="o",
        label="TCGA", **UCL_STYLE["TCGA"])
ax.fill_between(tcga["target_coverage"], tcga["automation_rate_lo"], tcga["automation_rate_hi"],
                 color=UCL_STYLE["TCGA"]["color"], alpha=0.15, lw=0)
ax.plot(ipd["target_coverage"], ipd["automation_rate"], marker="s",
        label="IPD Brain", **UCL_STYLE["IPD Brain"])
ax.fill_between(ipd["target_coverage"], ipd["automation_rate_lo"], ipd["automation_rate_hi"],
                 color=UCL_STYLE["IPD Brain"]["color"], alpha=0.15, lw=0)
ax.plot(tcga["target_coverage"], tcga["target_coverage"], ls=":", c="#888888", lw=1,
        label="per-stage target (reference)")
ax.set(xlabel="per-stage target coverage", ylabel="realised automation rate",
       title="Automation rate vs. coverage")
ax.invert_xaxis()
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "r6_pipeline_tcga_vs_ipd.pdf")
plt.savefig(FIG_DIR / "r6_pipeline_tcga_vs_ipd.png", dpi=150)
print("Saved to", FIG_DIR / "r6_pipeline_tcga_vs_ipd.pdf")
print("\nTCGA oracle-Stage-1 baseline:", tcga["full_set_accuracy"].iloc[0])
print("IPD Brain (confident) oracle-Stage-1 baseline:", ipd["full_set_accuracy"].iloc[0])
