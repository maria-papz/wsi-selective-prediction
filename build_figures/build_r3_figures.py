"""
R3a headline figure: retained AUROC and accuracy vs. coverage, ensemble
predictor, TCGA vs. IPD Brain. Standalone - reads only saved CSVs plus the
IPD Brain master table directly, no dependency on either notebook's
in-memory state.

"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

BASE = Path("/cs/student/project_msc/2025/aibh/mpapageo")
FIG_DIR = BASE / "outputs/r3_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

UCL_STYLE = {
    "TCGA":                    dict(color="#30d6ff", ls="-"),   # heritage blue, solid
    "IPD Brain":               dict(color="#ba82ff", ls="--"),  # mid purple, dashed
    "IPD Brain (no outliers)": dict(color="#993bff", ls=":"),   # bright purple, dotted
}

tcga = pd.read_csv(BASE / "outputs/selective_prediction_bench/selective_metrics_cal_locked_all_fdr.csv")
tcga_sub = tcga[(tcga.predictor == "ensemble") & (tcga.family == "predictive")]
tcga_auroc = tcga_sub[tcga_sub.metric == "auroc"].sort_values("target_coverage")
tcga_acc = tcga_sub[tcga_sub.metric == "accuracy"].sort_values("target_coverage")

ipd = pd.read_csv(BASE / "outputs/ipd_brain_baseline_bench/uncertainty_deferral_locked_vs_recalibrated.csv")
ipd_sub = ipd[(ipd.predictor == "ensemble") & (ipd.family == "predictive")].sort_values("target_coverage")


master = pd.read_csv(BASE / "outputs/ipd_brain_baseline_bench/master_patient_table.csv")
ipd_full_acc = ((master["prob_mutant"] >= 0.4416).astype(int) == master["label"]).mean()


curation = pd.read_csv(BASE / "outputs/ipd_brain_curation/patient_level_curation.csv")
curation["is_gbm_hist"] = curation["diagnosis"].str.contains("glioblastoma", case=False, na=False)
curation["is_grade4"] = curation["diagnosis"].str.contains(
    r"grade[\s\-]*(?:4|iv)\b", case=False, na=False, regex=True)
gbm_lookup = curation.set_index("patient_id")[["is_gbm_hist", "is_grade4"]]
gbm_or_g4 = (master["patient"].map(gbm_lookup["is_gbm_hist"]).fillna(False)
             | master["patient"].map(gbm_lookup["is_grade4"]).fillna(False)).to_numpy(bool)

y_ipd = master["label"].to_numpy(int)
p_ipd = master["prob_mutant"].to_numpy(float)
thr_ipd = 0.4416
pred_ipd = (p_ipd >= thr_ipd).astype(int)
is_err_ipd = pred_ipd != y_ipd
is_mut_ipd = y_ipd == 1

tcga_cutoffs = pd.read_csv(BASE / "outputs/selective_prediction_bench/selective_metrics_cal_locked_all_fdr.csv")
cutoff_lookup = (tcga_cutoffs[tcga_cutoffs.predictor == "ensemble"]
                 .drop_duplicates(["family", "target_coverage"])
                 .set_index(["family", "target_coverage"])[["signal", "calibration_cutoff"]])

no_outlier_rows = []
for cov in (0.9, 0.8, 0.7):
    sig, cutoff = cutoff_lookup.loc[("predictive", cov)]
    s = master[sig].to_numpy(float)
    keep = s <= float(cutoff)
    outliers = keep & is_mut_ipd & is_err_ipd & (np.abs(p_ipd - 0.5) > 0.3) & gbm_or_g4
    keep_no_out = keep & ~outliers
    no_outlier_rows.append({
        "target_coverage": cov,
        "auroc_no_outliers": roc_auc_score(y_ipd[keep_no_out], p_ipd[keep_no_out]),
        "acc_no_outliers": (pred_ipd[keep_no_out] == y_ipd[keep_no_out]).mean(),
        "n_outliers": int(outliers.sum()),
    })
ipd_no_outliers = pd.DataFrame(no_outlier_rows).sort_values("target_coverage")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

ax = axes[0]
ax.plot(tcga_auroc["target_coverage"], tcga_auroc["retained"], marker="o",
        label="TCGA", **UCL_STYLE["TCGA"])
ax.axhline(tcga_auroc["full"].iloc[0], color=UCL_STYLE["TCGA"]["color"], lw=0.8, alpha=0.4,
           label="TCGA (full cohort)")
ax.plot(ipd_sub["target_coverage"], ipd_sub["primary_retained_auroc"], marker="s",
        label="IPD Brain", **UCL_STYLE["IPD Brain"])
ax.axhline(ipd_sub["primary_full_auroc"].iloc[0], color=UCL_STYLE["IPD Brain"]["color"], lw=0.8, alpha=0.4,
           label="IPD Brain (full cohort)")
ax.plot(ipd_no_outliers["target_coverage"], ipd_no_outliers["auroc_no_outliers"], marker="^",
        label="IPD Brain (GBM-histology outliers excl.)", **UCL_STYLE["IPD Brain (no outliers)"])
ax.set(xlabel="target coverage", ylabel="retained AUROC", title="AUROC vs. coverage")
ax.invert_xaxis()
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(tcga_acc["target_coverage"], tcga_acc["retained"], marker="o",
        label="TCGA", **UCL_STYLE["TCGA"])
ax.axhline(tcga_acc["full"].iloc[0], color=UCL_STYLE["TCGA"]["color"], lw=0.8, alpha=0.4,
           label="TCGA (full cohort)")
ax.plot(ipd_sub["target_coverage"], ipd_sub["primary_retained_acc"], marker="s",
        label="IPD Brain", **UCL_STYLE["IPD Brain"])
ax.axhline(ipd_full_acc, color=UCL_STYLE["IPD Brain"]["color"], lw=0.8, alpha=0.4,
           label="IPD Brain (full cohort)")
ax.plot(ipd_no_outliers["target_coverage"], ipd_no_outliers["acc_no_outliers"], marker="^",
        label="IPD Brain (GBM-histology outliers excl.)", **UCL_STYLE["IPD Brain (no outliers)"])
ax.set(xlabel="target coverage", ylabel="retained accuracy", title="Accuracy vs. coverage")
ax.invert_xaxis()
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("Ensemble predictor, predictive family, TCGA-locked cutoffs -- "
             "faint horizontal lines are each cohort's no-abstention baseline", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "r3a_headline_tcga_vs_ipd.pdf")
plt.savefig(FIG_DIR / "r3a_headline_tcga_vs_ipd.png", dpi=150)
print("Saved to", FIG_DIR / "r3a_headline_tcga_vs_ipd.pdf")
print(f"IPD Brain full-cohort accuracy (locked threshold 0.4416): {ipd_full_acc:.4f}")
