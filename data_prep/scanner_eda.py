"""
Scanner EDA - final combined script.
Produces 4 figures:
  1. fig_scanner_race_minority.png   - % minority per scanner (3 subplots)
  2. fig_scanner_race_comparison.png - scanner distribution White vs Black/AA
  3. fig_magnification.png           - magnification overview (3 panels)
  4. fig_site_scanner_bar.png        - scanner diversity per site

Run from project root:
    source venv/bin/activate
    python scripts/plot_scanner_eda_final.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import chi2_contingency

OUT  = Path("outputs/scanner_metadata")
BASE = Path(".")
OUT.mkdir(parents=True, exist_ok=True)

#  load & merge 
df_scan = pd.read_csv(OUT / "tcga_scanner_metadata.csv")
df_clin = pd.read_csv(BASE / "data/raw/tcga/clinical/tcga_clinical.csv")
df_scan["case_id"] = df_scan["case_id"].str.strip()
df_clin["case_id"] = df_clin["case_id"].str.strip()

df = df_scan.merge(
    df_clin[["case_id", "idh_status", "demographic.race",
             "tissue_source_site.code", "tissue_source_site.name",
             "project.project_id"]],
    on="case_id", how="left"
)
df["model"] = df["model"].fillna("").str.strip().replace("", "Unknown")
df["mag"]   = df["objective_power"].fillna(0).astype(int)

race_map = {
    "white":                            "White",
    "black or african american":        "Black/AA",
    "asian":                            "Asian",
    "american indian or alaska native": "Native Am.",
    "not reported":                     "Not reported",
    "not allowed to collect":           "Not reported",
}
df["race_group"] = df["demographic.race"].str.lower().map(race_map).fillna("Other/Unknown")
df["site_label"] = (df["tissue_source_site.code"].fillna("") + "  " +
                    df["tissue_source_site.name"].fillna(""))

top_models   = (df[df["model"] != "Unknown"]["model"]
                .value_counts().head(10).index.tolist())
df_top       = df[df["model"].isin(top_models)].copy()
model_order  = df_top["model"].value_counts().index.tolist()

BLUE  = "#2166ac"
RED   = "#d6604d"
GREY  = "#aaaaaa"

# 
# FIGURE 1: % minority per scanner  3 subplots
# 
minority_races = ["Black/AA", "Asian", "Native Am."]
palette = {"Black/AA": "#d6604d", "Asian": "#4393c3", "Native Am.": "#f4a582"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, race in zip(axes, minority_races):
    pcts, ns = [], []
    for model in model_order:
        sub    = df_top[df_top["model"] == model]
        n_race = (sub["race_group"] == race).sum()
        pcts.append(n_race / len(sub) * 100 if len(sub) > 0 else 0)
        ns.append(n_race)

    bars = ax.bar(range(len(model_order)), pcts,
                  color=palette[race], edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("% of scanner's patients", fontsize=10)
    ax.set_title(f"{race} patients\nper scanner", fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    for bar, n in zip(bars, ns):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.15,
                    str(n), ha="center", fontsize=8, color="#333333")

    overall_pct = (df_top["race_group"] == race).mean() * 100
    ax.axhline(overall_pct, color="black", linewidth=1.2, linestyle="--", alpha=0.6,
               label=f"Overall avg ({overall_pct:.1f}%)")
    ax.legend(fontsize=8, loc="upper right")

plt.suptitle("Minority race representation per scanner model\n(dashed = cohort average)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig_scanner_race_minority.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_scanner_race_minority.png")


# 
# FIGURE 2: scanner distribution White vs Black/AA
# 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, race in zip(axes, ["White", "Black/AA"]):
    sub    = df_top[df_top["race_group"] == race]
    counts = sub["model"].value_counts().reindex(model_order, fill_value=0)
    pcts   = counts / counts.sum() * 100
    color  = BLUE if race == "White" else RED

    bars = ax.bar(range(len(model_order)), pcts,
                  color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("% of group's slides", fontsize=10)
    ax.set_title(f"{race} patients (n={len(sub)})\nScanner distribution",
                 fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    for bar, n in zip(bars, counts.values):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    str(n), ha="center", fontsize=7.5, color="#333333")

plt.suptitle("Scanner usage: White vs Black/AA patients",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig_scanner_race_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_scanner_race_comparison.png")

# chi-square
contingency = pd.crosstab(df_top["model"], df_top["race_group"])
chi2, p, dof, _ = chi2_contingency(contingency)
print(f"Chi-square (race × scanner): chi2={chi2:.2f}, p={p:.4f}")


# 
# FIGURE 3: Magnification overview  3 panels
# 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: magnification × IDH
mag_idh = df.groupby(["mag", "idh_status"]).size().unstack(fill_value=0)
mag_idh = mag_idh.reindex([20, 40]).fillna(0)
x = np.arange(2)
w = 0.3
b1 = axes[0].bar(x - w/2, mag_idh.get("Mutant",   pd.Series([0,0], index=[20,40])),
                 width=w, color=BLUE, label="IDH-mutant")
b2 = axes[0].bar(x + w/2, mag_idh.get("Wildtype", pd.Series([0,0], index=[20,40])),
                 width=w, color=RED,  label="IDH-wildtype")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["20x", "40x"], fontsize=11)
axes[0].set_ylabel("Number of slides", fontsize=10)
axes[0].set_title("Magnification × IDH status", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
for bar in list(b1) + list(b2):
    v = int(bar.get_height())
    if v > 0:
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 1, str(v),
                     ha="center", fontsize=8)
axes[0].spines[["top","right"]].set_visible(False)

# Panel B: MPP histogram split by magnification
mpp_20 = df[df["mag"] == 20]["mpp_x"].dropna()
mpp_40 = df[df["mag"] == 40]["mpp_x"].dropna()
axes[1].hist(mpp_40, bins=20, color=BLUE, alpha=0.7, label="40x slides")
axes[1].hist(mpp_20, bins=20, color=RED,  alpha=0.7, label="20x slides")
axes[1].axvline(0.50, color=RED,  linewidth=1.5, linestyle="--", alpha=0.8, label="0.5 µm/px (20x target)")
axes[1].axvline(0.25, color=BLUE, linewidth=1.5, linestyle="--", alpha=0.8, label="0.25 µm/px (40x)")
axes[1].set_xlabel("MPP (µm/pixel)", fontsize=10)
axes[1].set_ylabel("Number of slides", fontsize=10)
axes[1].set_title("MPP distribution by magnification", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=8)
axes[1].spines[["top","right"]].set_visible(False)

# Panel C: magnification by project
mag_proj = df.groupby(["project.project_id", "mag"]).size().unstack(fill_value=0)
mag_proj = mag_proj.reindex(["TCGA-GBM", "TCGA-LGG"]).fillna(0)
b3 = axes[2].bar(x - w/2, mag_proj.get(20, pd.Series([0,0], index=["TCGA-GBM","TCGA-LGG"])),
                 width=w, color=RED,  label="20x")
b4 = axes[2].bar(x + w/2, mag_proj.get(40, pd.Series([0,0], index=["TCGA-GBM","TCGA-LGG"])),
                 width=w, color=BLUE, label="40x")
axes[2].set_xticks(x)
axes[2].set_xticklabels(["TCGA-GBM", "TCGA-LGG"], fontsize=10)
axes[2].set_ylabel("Number of slides", fontsize=10)
axes[2].set_title("Magnification by TCGA project", fontsize=11, fontweight="bold")
axes[2].legend(fontsize=9)
for bar in list(b3) + list(b4):
    v = int(bar.get_height())
    if v > 0:
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 1, str(v),
                     ha="center", fontsize=8)
axes[2].spines[["top","right"]].set_visible(False)

plt.suptitle("TCGA scan resolution heterogeneity", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "fig_magnification.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_magnification.png")


# 
# FIGURE 4: scanner diversity per site  bar chart
# 
site_stats = (df.groupby("site_label")
              .agg(n_slides=("model","count"), n_scanners=("model","nunique"))
              .reset_index())

plot_df = (site_stats[site_stats["n_slides"] >= 5]
           .sort_values("n_scanners", ascending=False)
           .reset_index(drop=True))

color_map = {1: GREY, 2: "#fdae61", 3: "#d73027"}
colors = [color_map.get(n, "#a50026") for n in plot_df["n_scanners"]]

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(range(len(plot_df)), plot_df["n_scanners"],
              color=colors, edgecolor="white", linewidth=0.4)

ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels(plot_df["site_label"], rotation=55, ha="right", fontsize=7.5)
ax.set_ylabel("Number of distinct scanner models", fontsize=11)
ax.set_title("Scanner heterogeneity per tissue source site (sites 5 slides)",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, plot_df["n_scanners"].max() + 1)
ax.spines[["top","right"]].set_visible(False)

for bar, row in zip(bars, plot_df.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"n={row.n_slides}", ha="center", va="bottom",
            fontsize=6.5, color="#444444")

legend_handles = [mpatches.Patch(color=color_map.get(k,"#a50026"),
                                 label=f"{k} scanner{'s' if k>1 else ''}")
                  for k in [1, 2, 3]]
legend_handles.append(mpatches.Patch(color="#a50026", label="4+ scanners"))
ax.legend(handles=legend_handles, fontsize=9, title="Scanners per site", loc="upper right")

plt.tight_layout()
fig.savefig(OUT / "fig_site_scanner_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig_site_scanner_bar.png")

print("\nAll done. Figures saved to outputs/scanner_metadata/")