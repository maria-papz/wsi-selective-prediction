# scripts/analyse_ipd_brain.py
"""
IPD-Brain (Chauhan et al., 2024) EDA -- publication-quality figures.
Input:  data/raw/ipd_brain/clinical/ipd_brain_v1.csv       (raw, 328 patients)
        outputs/ipd_brain_curation/patient_level_curation.csv (curated cohort, from
                                                                 check_ipd_brain_exclusions.py)
        outputs/ipd_brain_curation/slide_level_curation.csv    (kept-slide file sizes)
Output: outputs/ipd_brain_eda/*.png, outputs/ipd_brain_eda/summary_table.csv
"""

import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu

warnings.filterwarnings("ignore")
pd.set_option("future.no_silent_downcasting", True)

OUT = Path("outputs/ipd_brain_eda")
OUT.mkdir(parents=True, exist_ok=True)

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

MUT  = "#C0392B"
WT   = "#2471A3"
GREY = "#7F8C8D"


def save(name):
    plt.savefig(OUT / f"{name}.png")
    plt.close()
    print(f"   {name}.png")


# -- load & join curation output with raw clinical fields ---------------------
SLIDE_SUFFIX_RE = re.compile(r"\([a-z]\)$", re.IGNORECASE)

raw = pd.read_csv("data/raw/ipd_brain/clinical/ipd_brain_v1.csv", encoding="utf-8-sig")
raw["patient_id"] = raw["Case Number"].astype(str).str.split("\n").str[0].str.strip()
raw["patient_id"] = raw["patient_id"].str.replace(SLIDE_SUFFIX_RE, "", regex=True)

patient_curation = pd.read_csv("outputs/ipd_brain_curation/patient_level_curation.csv")
slide_curation = pd.read_csv("outputs/ipd_brain_curation/slide_level_curation.csv")

included_ids = patient_curation.loc[patient_curation["status"] == "included", "patient_id"]
df = raw[raw["patient_id"].isin(included_ids)].copy()

kept_sizes = (
    slide_curation[slide_curation["status"] == "included"]
    .set_index("patient_id")["file_size_mb"]
)
df["size_mb"] = df["patient_id"].map(kept_sizes)

df["idh_status"] = df["IDH1R132H"].map({1: "mutant", 0: "wildtype"})
df["sex"] = df["Sex"].map({"M": "male", "F": "female"})
df["subtype"] = df["Subtype"].str.title()
df["atrx"] = df["ATRX"].replace({"Not Retained": "lost", "retained": "retained"})
df["p53_status"] = df["p53"].map({1: "positive", 0: "negative"})
df["grade"] = df["WHO Grade"].astype("Int64")

LOBE_KEYWORDS = [
    ("frontal", "Frontal"),
    ("temporal", "Temporal"),
    ("parietal", "Parietal"),
    ("occipital", "Occipital"),
    ("insular", "Insular"),
    ("corpus", "Corpus callosum"),
    ("cerebell", "Cerebellum"),
    ("thalam", "Thalamus"),
    ("ventric", "Ventricle"),
]


def normalise_site(raw_site):
    s = str(raw_site).lower()
    for kw, label in LOBE_KEYWORDS:
        if kw in s:
            return label
    return "Other / unspecified"


df["site_grouped"] = df["SITE"].apply(normalise_site)

print(f"Loaded {len(df)} curated cases (from {len(raw)} in raw CSV)")

#
# FIG 1 -- Cohort overview (3 panels: IDH by subtype / sex / grade)
#
print("\nFig 1  Cohort overview")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

# 1a IDH by subtype
ax = axes[0]
subtypes = ["Glioblastoma", "Astrocytoma", "Oligodendroglioma"]
wt_n  = [len(df[(df["subtype"] == t) & (df["idh_status"] == "wildtype")]) for t in subtypes]
mut_n = [len(df[(df["subtype"] == t) & (df["idh_status"] == "mutant")]) for t in subtypes]
x = np.arange(len(subtypes))
ax.bar(x, wt_n,  width=0.5, color=WT,  label="IDH-wildtype", zorder=3)
ax.bar(x, mut_n, width=0.5, color=MUT, label="IDH-mutant", bottom=wt_n, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(["GBM", "Astro", "Oligo"])
ax.set_ylabel("Cases"); ax.set_title("(a) IDH status by subtype")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
ax.legend(frameon=False, fontsize=7)
for i, (w, m) in enumerate(zip(wt_n, mut_n)):
    t = w + m
    ax.text(i, t + 3, str(t), ha="center", fontsize=8, fontweight="bold")
    if w:
        ax.text(i, w / 2, str(w), ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    if m:
        ax.text(i, w + m / 2, str(m), ha="center", va="center", color="white", fontsize=7, fontweight="bold")

# 1b sex
ax = axes[1]
sc = df["sex"].value_counts()
scols = {"male": "#2980B9", "female": "#E74C3C"}
bars = ax.barh(sc.index, sc.values, color=[scols.get(s, GREY) for s in sc.index],
               edgecolor="white", lw=0.5, zorder=3)
ax.set_xlabel("Cases"); ax.set_title("(b) Sex")
ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
for bar, v in zip(bars, sc.values):
    ax.text(v + 2, bar.get_y() + bar.get_height() / 2,
            f"{v} ({100 * v / len(df):.0f}%)", va="center", fontsize=7.5)

# 1c WHO grade
ax = axes[2]
gc = df["grade"].value_counts().sort_index()
bars = ax.bar(gc.index.astype(str), gc.values, color=GREY, edgecolor="white", lw=0.5, zorder=3)
ax.set_xlabel("WHO Grade"); ax.set_ylabel("Cases"); ax.set_title("(c) WHO Grade")
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
for bar, v in zip(bars, gc.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 2, str(v), ha="center", fontsize=8)

plt.suptitle(f"IPD-Brain Cohort Overview  (n={len(df)}, one slide per patient)",
             fontsize=10, fontweight="bold", y=1.02)
plt.tight_layout()
save("fig1_cohort_overview")

#
# FIG 2 -- Age distribution by IDH status
#
print("Fig 2  Age distribution")
fig, ax = plt.subplots(figsize=(6, 3.5))

mut_ages = df[df["idh_status"] == "mutant"]["Age"].dropna()
wt_ages  = df[df["idh_status"] == "wildtype"]["Age"].dropna()
_, p_age = mannwhitneyu(mut_ages, wt_ages, alternative="two-sided")

mut_ages.plot.kde(ax=ax, color=MUT, linewidth=2,
                   label=f"IDH-mutant (n={len(mut_ages)})", zorder=3)
wt_ages.plot.kde(ax=ax, color=WT, linewidth=2,
                  label=f"IDH-wildtype (n={len(wt_ages)})", zorder=3)
ax.set_xlim(df["Age"].min() - 5, df["Age"].max() + 5)
ax.set_ylim(bottom=0)
ax.axvline(mut_ages.median(), color=MUT, ls="--", lw=1.5, alpha=0.9)
ax.axvline(wt_ages.median(),  color=WT,  ls="--", lw=1.5, alpha=0.9)
ax.text(mut_ages.median() - 1, ax.get_ylim()[1] * 0.92, f"Median\n{mut_ages.median():.0f}y",
        ha="right", color=MUT, fontsize=7.5, fontweight="bold")
ax.text(wt_ages.median() + 1, ax.get_ylim()[1] * 0.92, f"Median\n{wt_ages.median():.0f}y",
        ha="left", color=WT, fontsize=7.5, fontweight="bold")
ax.set_xlabel("Age at diagnosis (years)")
ax.set_ylabel("Density")
ax.set_title(f"Age Distribution by IDH Status\nMann-Whitney U  p = {p_age:.1e}")
ax.legend(frameon=False)
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
plt.tight_layout()
save("fig2_age_distribution")

#
# FIG 3 -- WHO Grade by IDH status
#
print("Fig 3  Grade by IDH status")
grade_ct = df.groupby(["grade", "idh_status"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(grade_ct.index.astype(str), grade_ct.get("wildtype", 0), width=0.5,
       color=WT, label="IDH-wildtype", zorder=3, edgecolor="white")
ax.bar(grade_ct.index.astype(str), grade_ct.get("mutant", 0), width=0.5,
       color=MUT, label="IDH-mutant", zorder=3, edgecolor="white",
       bottom=grade_ct.get("wildtype", 0))
ax.set_xlabel("WHO Grade"); ax.set_ylabel("Cases")
ax.set_title("WHO Grade by IDH Status")
ax.legend(frameon=False); ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
for container in ax.containers:
    ax.bar_label(container, padding=2, fontsize=7)
plt.tight_layout()
save("fig3_grade_by_idh")

#
# FIG 4 -- ki67 proliferation index by IDH status
#
print("Fig 4  ki67 by IDH status")
mut_ki = df[df["idh_status"] == "mutant"]["ki67(in %)"].dropna()
wt_ki  = df[df["idh_status"] == "wildtype"]["ki67(in %)"].dropna()
_, p_ki = mannwhitneyu(mut_ki, wt_ki, alternative="two-sided")

fig, ax = plt.subplots(figsize=(6, 3.5))
bp = ax.boxplot([wt_ki, mut_ki], tick_labels=["IDH-wildtype", "IDH-mutant"],
                patch_artist=True, widths=0.5, flierprops={"markersize": 3})
for patch, color in zip(bp["boxes"], [WT, MUT]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel("ki67 (%)")
ax.set_title(f"Ki67 Proliferation Index by IDH Status\nMann-Whitney U  p = {p_ki:.1e}")
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
plt.tight_layout()
save("fig4_ki67_by_idh")

#
# FIG 5 -- ATRX status by IDH status
#
print("Fig 5  ATRX by IDH status")
atrx_ct = pd.crosstab(df["atrx"], df["idh_status"])
_, p_atrx, _, _ = chi2_contingency(atrx_ct)
fig, ax = plt.subplots(figsize=(5, 4))
x = np.arange(len(atrx_ct))
ax.bar(x, atrx_ct.get("wildtype", 0), width=0.5, color=WT, label="IDH-wildtype",
       zorder=3, edgecolor="white")
ax.bar(x, atrx_ct.get("mutant", 0), width=0.5, color=MUT, label="IDH-mutant",
       zorder=3, edgecolor="white", bottom=atrx_ct.get("wildtype", 0))
ax.set_xticks(x); ax.set_xticklabels(atrx_ct.index)
ax.set_ylabel("Cases")
ax.set_title(f"ATRX Status by IDH Status\n"
             f"$\\chi^2$ p = {p_atrx:.1e}")
ax.legend(frameon=False); ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
for container in ax.containers:
    ax.bar_label(container, padding=2, fontsize=8)
plt.tight_layout()
save("fig5_atrx_by_idh")

#
# FIG 6 -- p53 status by IDH status
#
print("Fig 6  p53 by IDH status")
p53_ct = pd.crosstab(df["p53_status"], df["idh_status"])
_, p_p53, _, _ = chi2_contingency(p53_ct)
fig, ax = plt.subplots(figsize=(5, 4))
x = np.arange(len(p53_ct))
ax.bar(x, p53_ct.get("wildtype", 0), width=0.5, color=WT, label="IDH-wildtype",
       zorder=3, edgecolor="white")
ax.bar(x, p53_ct.get("mutant", 0), width=0.5, color=MUT, label="IDH-mutant",
       zorder=3, edgecolor="white", bottom=p53_ct.get("wildtype", 0))
ax.set_xticks(x); ax.set_xticklabels(p53_ct.index)
ax.set_ylabel("Cases")
ax.set_title(f"p53 Status by IDH Status\n"
             f"$\\chi^2$ p = {p_p53:.1e}")
ax.legend(frameon=False); ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
for container in ax.containers:
    ax.bar_label(container, padding=2, fontsize=8)
plt.tight_layout()
save("fig6_p53_by_idh")

#
# FIG 7 -- Anatomical site (grouped) by IDH status
#
print("Fig 7  Site by IDH status")
site_counts = df["site_grouped"].value_counts()
keep = site_counts[site_counts >= 5].index
site_ct = (
    df[df["site_grouped"].isin(keep)]
    .groupby(["site_grouped", "idh_status"]).size().unstack(fill_value=0)
    .assign(total=lambda d: d.sum(axis=1))
    .sort_values("total", ascending=False)
    .drop(columns="total")
)
fig, ax = plt.subplots(figsize=(8, 4.5))
site_ct.plot(kind="bar", ax=ax, color=[WT, MUT], edgecolor="white", width=0.65, zorder=3)
ax.set_title("Anatomical Site by IDH Status\n(sites with ≥5 cases)", fontweight="bold")
ax.set_ylabel("Cases"); ax.set_xlabel("")
ax.set_xticklabels(site_ct.index, rotation=30, ha="right")
ax.legend(["IDH-wildtype", "IDH-mutant"], frameon=False)
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
plt.tight_layout()
save("fig7_site_by_idh")

#
# FIG 8 -- Slide file sizes
#
print("Fig 8  Slide sizes")
fig, ax = plt.subplots(figsize=(6, 3.5))
sizes = df["size_mb"].dropna()
ax.hist(sizes, bins=40, color="#1A5276", edgecolor="white", lw=0.3, alpha=0.85, zorder=3)
ax.axvline(sizes.median(), color=MUT, ls="--", lw=1.5, label=f"Median: {sizes.median():.0f} MB")
ax.set_xlabel("File size (MB)")
ax.set_ylabel("Number of slides")
ax.set_title(f"Kept-Slide File Sizes  (n={len(sizes)} slides, "
             f"total {sizes.sum() / 1e3:.0f} GB)")
ax.legend(frameon=False)
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
plt.tight_layout()
save("fig8_slide_sizes")

#
# FIG 9 -- Age vs ki67, coloured by IDH status (relationship between the two
# continuous fairness/biology-relevant covariates)
#
print("Fig 9  Age vs ki67 scatter")
fig, ax = plt.subplots(figsize=(6, 4.5))
for label, color in [("wildtype", WT), ("mutant", MUT)]:
    sub = df[df["idh_status"] == label]
    ax.scatter(sub["Age"], sub["ki67(in %)"], s=14, color=color, alpha=0.6,
               edgecolor="white", lw=0.3, label=f"IDH-{label}", zorder=3)
ax.set_xlabel("Age at diagnosis (years)")
ax.set_ylabel("ki67 (%)")
ax.set_title("Age vs Ki67 Proliferation Index by IDH Status")
ax.legend(frameon=False)
ax.grid(lw=0.4, alpha=0.5, zorder=0)
plt.tight_layout()
save("fig9_age_vs_ki67")

#
# Summary table
#
print("\nGenerating summary CSV...")


def make_row(stratum, group, sub):
    n = len(sub)
    n_mut = (sub["idh_status"] == "mutant").sum()
    n_wt = (sub["idh_status"] == "wildtype").sum()
    ages = sub["Age"].dropna()
    ki = sub["ki67(in %)"].dropna()
    return {
        "stratum": stratum,
        "group": group,
        "n": n,
        "n_idh_mutant": n_mut,
        "n_idh_wildtype": n_wt,
        "pct_idh_mutant": f"{100 * n_mut / n:.1f}%" if n > 0 else "-",
        "pct_idh_wildtype": f"{100 * n_wt / n:.1f}%" if n > 0 else "-",
        "age_median": f"{ages.median():.0f}" if len(ages) > 0 else "-",
        "ki67_median": f"{ki.median():.0f}" if len(ki) > 0 else "-",
        "pct_male": f"{100 * (sub['sex'] == 'male').mean():.1f}%" if n > 0 else "-",
    }


rows = [make_row("Overall", "All curated cases", df)]

for subtype in ["Glioblastoma", "Astrocytoma", "Oligodendroglioma"]:
    rows.append(make_row("Subtype", subtype, df[df["subtype"] == subtype]))

for grade in sorted(df["grade"].dropna().unique()):
    rows.append(make_row("WHO Grade", f"Grade {grade}", df[df["grade"] == grade]))

for sex_val, sex_label in [("male", "Male"), ("female", "Female")]:
    rows.append(make_row("Sex", sex_label, df[df["sex"] == sex_val]))

for atrx_val in df["atrx"].dropna().unique():
    rows.append(make_row("ATRX", atrx_val, df[df["atrx"] == atrx_val]))

for p53_val in df["p53_status"].dropna().unique():
    rows.append(make_row("p53", p53_val, df[df["p53_status"] == p53_val]))

for site in site_counts.index:
    rows.append(make_row("Site", site, df[df["site_grouped"] == site]))

df_summary = pd.DataFrame(rows)
df_summary.to_csv(OUT / "summary_table.csv", index=False)
print(f"   summary_table.csv  ({len(df_summary)} rows)")
print(df_summary.head(10).to_string(index=False))

print(f"\nDone -- all figures saved to {OUT}/")
