# scripts/analyse_tcga.py
"""
TCGA-GBM/LGG EDA 
Input:  data/raw/tcga/clinical/tcga_manifest.csv  (789 usable cases, deduped)
        data/raw/tcga/clinical/tcga_clinical.csv   (for codel subtype)
Output: outputs/tcga_eda/*.png
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu, chi2_contingency
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

OUT = Path("outputs/tcga_eda")
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
GBM  = "#1A5276"
LGG  = "#148F77"
GREY = "#7F8C8D"

def save(name):
    plt.savefig(OUT / f"{name}.png")
    plt.close()
    print(f"   {name}.png")

#  load 
df = pd.read_csv("data/raw/tcga/clinical/tcga_manifest.csv")
df["age"]  = pd.to_numeric(df["age"], errors="coerce")
df["race"] = df["race"].fillna("not reported").str.lower()
df["sex"]  = df["sex"].fillna("not reported").str.lower()
df["site"] = df["site"].fillna("Unknown")

# merge codel + tissue origin from clinical table
df_clin = pd.read_csv("data/raw/tcga/clinical/tcga_clinical.csv")
df_clin["case_id"] = df_clin["submitter_id"].str[:12]

origin_cols = [c for c in df_clin.columns if "tissue_or_organ_of_origin" in c]
df_clin["tissue_origin"] = df_clin[origin_cols].bfill(axis=1).iloc[:, 0]

df = df.merge(
    df_clin[["case_id", "idh_codel", "tissue_origin"]].drop_duplicates("case_id"),
    on="case_id", how="left"
)
df["brain_region"] = (
    df["tissue_origin"]
    .fillna("Not reported")
    .str.replace(r"Brain,\s*", "", regex=True)
    .str.strip()
    .str.title()
    .replace("", "Not reported")
)

VALID_BRAIN = {
    "brain, nos":                    "Brain NOS",
    "cerebrum":                      "Cerebrum",
    "temporal lobe":                 "Temporal lobe",
    "frontal lobe":                  "Frontal lobe",
    "occipital lobe":                "Occipital lobe",
    "parietal lobe":                 "Parietal lobe",
    "cerebellum":                    "Cerebellum",
    "brainstem":                     "Brainstem",
    "overlapping lesion of brain":   "Overlapping lesion",
    "nervous system, nos":           "Nervous system NOS",
}

def clean_region(raw):
    if pd.isna(raw) or str(raw).strip().lower() == "not reported":
        return "Not reported"
    key = str(raw).strip().lower()
    return VALID_BRAIN.get(key, "Not reported")  # anything not in whitelist ? Not reported

df["brain_region"] = df["tissue_origin"].apply(clean_region)

print(f"Loaded {len(df)} usable cases")

# 
# FIG 1  Cohort overview (3 panels: IDH by project / sex / race)
# 
print("\nFig 1  Cohort overview")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

# 1a IDH by project
ax = axes[0]
projs = ["TCGA-GBM", "TCGA-LGG"]
wt_n  = [len(df[(df["project"]==p) & (df["idh_status"]=="wildtype")]) for p in projs]
mut_n = [len(df[(df["project"]==p) & (df["idh_status"]=="mutant")])   for p in projs]
x = np.arange(2)
ax.bar(x, wt_n,  width=0.5, color=WT,  label="IDH-wildtype", zorder=3)
ax.bar(x, mut_n, width=0.5, color=MUT, label="IDH-mutant", bottom=wt_n, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(["GBM", "LGG"])
ax.set_ylabel("Cases"); ax.set_title("(a) IDH status by project")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
ax.legend(frameon=False, fontsize=7)
for i, (w, m) in enumerate(zip(wt_n, mut_n)):
    t = w + m
    ax.text(i, t+3, str(t), ha="center", fontsize=8, fontweight="bold")
    ax.text(i, w/2,   str(w), ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    ax.text(i, w+m/2, str(m), ha="center", va="center", color="white", fontsize=7, fontweight="bold")

# 1b sex
ax = axes[1]
sc = df["sex"].replace({"not reported":"unknown"}).value_counts()
sc = sc[sc.index.isin(["male","female","unknown"])]
scols = {"male":"#2980B9","female":"#E74C3C","unknown":GREY}
bars = ax.barh(sc.index, sc.values,
               color=[scols.get(s, GREY) for s in sc.index],
               edgecolor="white", lw=0.5, zorder=3)
ax.set_xlabel("Cases"); ax.set_title("(b) Sex")
ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
for bar, v in zip(bars, sc.values):
    ax.text(v+2, bar.get_y()+bar.get_height()/2,
            f"{v} ({100*v/len(df):.0f}%)", va="center", fontsize=7.5)

# 1c race
ax = axes[2]
rmap = {"white":"White","black or african american":"Black/AA",
        "asian":"Asian","american indian or alaska native":"AIAN",
        "not reported":"Not reported","unknown":"Unknown"}
rc   = df["race"].map(lambda x: rmap.get(x,"Other")).value_counts()
rcols = ["#2C3E50","#E74C3C","#F39C12","#27AE60","#95A5A6","#BDC3C7"]
bars = ax.barh(rc.index, rc.values,
               color=rcols[:len(rc)], edgecolor="white", lw=0.5, zorder=3)
ax.set_xlabel("Cases"); ax.set_title("(c) Race")
ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
for bar, v in zip(bars, rc.values):
    ax.text(v+1, bar.get_y()+bar.get_height()/2,
            f"{v} ({100*v/len(df):.0f}%)", va="center", fontsize=7)

plt.suptitle(f"TCGA-GBM/LGG Cohort Overview  (n={len(df)}, one DX slide per patient)",
             fontsize=10, fontweight="bold", y=1.02)
plt.tight_layout()
save("fig1_cohort_overview")

# 
# FIG 2  Age distribution (one curve per IDH status, no project split)
# 
print("Fig 2  Age distribution")
fig, ax = plt.subplots(figsize=(6, 3.5))

mut_ages = df[df["idh_status"]=="mutant"]["age"].dropna()
wt_ages  = df[df["idh_status"]=="wildtype"]["age"].dropna()
_, p_age = mannwhitneyu(mut_ages, wt_ages, alternative="two-sided")

bins = np.linspace(df["age"].min(), df["age"].max(), 30)
ax.hist(mut_ages, bins=bins, alpha=0.7, color=MUT, edgecolor="white",
        lw=0.4, label=f"IDH-mutant (n={len(mut_ages)})", zorder=3)
ax.hist(wt_ages,  bins=bins, alpha=0.7, color=WT,  edgecolor="white",
        lw=0.4, label=f"IDH-wildtype (n={len(wt_ages)})", zorder=3)
ax.axvline(mut_ages.median(), color=MUT, ls="--", lw=1.5, alpha=0.9)
ax.axvline(wt_ages.median(),  color=WT,  ls="--", lw=1.5, alpha=0.9)

ax.text(mut_ages.median()-1, ax.get_ylim()[1]*0.92,
        f"Median\n{mut_ages.median():.0f}y",
        ha="right", color=MUT, fontsize=7.5, fontweight="bold")
ax.text(wt_ages.median()+1, ax.get_ylim()[1]*0.92,
        f"Median\n{wt_ages.median():.0f}y",
        ha="left", color=WT, fontsize=7.5, fontweight="bold")

ax.set_xlabel("Age at diagnosis (years)")
ax.set_ylabel("Number of cases")
ax.set_title(f"Age Distribution by IDH Status\n"
             f"Mann-Whitney U  p = {p_age:.1e}")
ax.legend(frameon=False)
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
plt.tight_layout()
save("fig2_age_distribution")

# 
# FIG 3  Tissue source sites (top 15)
# 
print("Fig 3  Tissue source sites")
fig, ax = plt.subplots(figsize=(9, 5.5))

sc = df["site"].value_counts().head(15)
colors = [MUT if (df[df["site"]==s]["idh_status"]=="mutant").mean() >= 0.5
          else WT for s in sc.index]
bars = ax.barh(range(len(sc)), sc.values, color=colors,
               edgecolor="white", lw=0.5, zorder=3)
ax.set_yticks(range(len(sc)))
ax.set_yticklabels(sc.index, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Usable cases")
ax.set_title("Top 15 Tissue Source Sites  "
             "(red = IDH-mutant majority, blue = IDH-wildtype majority)", fontsize=9)
ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
for i, (bar, v) in enumerate(zip(bars, sc.values)):
    site = sc.index[i]
    sub  = df[df["site"]==site]
    nm   = (sub["idh_status"]=="mutant").sum()
    nw   = (sub["idh_status"]=="wildtype").sum()
    ax.text(v+0.5, bar.get_y()+bar.get_height()/2,
            f"n={v}  ({nm}M / {nw}W)", va="center", fontsize=7)
ax.legend(handles=[
    mpatches.Patch(color=MUT, label="IDH-mutant majority"),
    mpatches.Patch(color=WT,  label="IDH-wildtype majority"),
], frameon=False, loc="lower right")
plt.tight_layout()
save("fig3_tissue_source_sites")

# 
# FIG 4  Slide file sizes (simple: total size annotation, histogram only)
# 
print("Fig 4  Slide sizes")
fig, ax = plt.subplots(figsize=(6, 3.5))

ax.hist(df["size_mb"], bins=40, color=GBM, edgecolor="white",
        lw=0.3, alpha=0.85, zorder=3)
ax.axvline(df["size_mb"].median(), color=MUT, ls="--", lw=1.5,
           label=f"Median: {df['size_mb'].median():.0f} MB")
ax.set_xlabel("File size (MB)")
ax.set_ylabel("Number of slides")
ax.set_title(f"DX Slide File Sizes  (n={len(df)} slides, "
             f"total {df['size_mb'].sum()/1e3:.0f} GB)")
ax.legend(frameon=False)
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
plt.tight_layout()
save("fig4_slide_sizes")

# 
# FIG 5  WHO 2021 tumour types (single unified pie)
# 
print("Fig 5  WHO 2021 tumour types")
fig, ax = plt.subplots(figsize=(6, 5))

tcols = {"glioblastoma":GBM, "astrocytoma":MUT,
         "oligodendroglioma":LGG, "unclassified":GREY}

types = df["tumour_type_who2021"].fillna("unclassified").str.lower().value_counts()
cols  = [tcols.get(t, GREY) for t in types.index]

wedges, _, autos = ax.pie(
    types.values, colors=cols, labels=None,
    autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
    pctdistance=0.72, startangle=90,
    wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
)
for at in autos:
    at.set_fontsize(9); at.set_color("white"); at.set_fontweight("bold")

ax.legend(wedges,
          [f"{t.capitalize()} (n={v}, {100*v/len(df):.0f}%)"
           for t, v in zip(types.index, types.values)],
          loc="lower center", bbox_to_anchor=(0.5, -0.12),
          frameon=False, fontsize=9, ncol=2)

ax.set_title("WHO 2021 Tumour Type Classification\n", fontsize=10, fontweight="bold")
plt.tight_layout()
save("fig5_tumour_types_who2021")

# 
# FIG 6  IDH/codel subtype
# 
print("Fig 6  IDH/codel subtypes")
fig, ax = plt.subplots(figsize=(7, 3.5))

codel_order  = ["IDHwt", "IDHmut-non-codel", "IDHmut-codel"]
codel_labels = ["IDH-wildtype", "IDHmut non-codel\n(astrocytoma)",
                "IDHmut codel\n(oligodendroglioma)"]
codel_colors = [WT, "#E74C3C", LGG]

cc = df["idh_codel"].fillna("IDHwt").value_counts()
cc = cc.reindex([k for k in codel_order if k in cc.index])

bars = ax.bar(np.arange(len(cc)), cc.values,
              color=[codel_colors[codel_order.index(k)] for k in cc.index],
              edgecolor="white", lw=0.8, width=0.5, zorder=3)
ax.set_xticks(np.arange(len(cc)))
ax.set_xticklabels([codel_labels[codel_order.index(k)] for k in cc.index], fontsize=9)
ax.set_ylabel("Usable cases")
ax.set_title("IDH/1p19q Codeletion Subtype")
ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
ax.set_ylim(0, cc.max()*1.18)
for bar, v in zip(bars, cc.values):
    ax.text(bar.get_x()+bar.get_width()/2, v+4,
            f"n={v}\n({100*v/len(df):.0f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.tight_layout()
save("fig6_idh_codel_subtypes")

# 
# FIG 7  Race × IDH fairness
# 
print("Fig 7  Race × IDH fairness")

known     = df[~df["race"].isin(["not reported","unknown"])].copy()
known_bin = known["race"].apply(lambda r: "White" if r=="white" else "Other")
ct        = pd.crosstab(known_bin, known["idh_status"])
_, p_race, _, _ = chi2_contingency(ct)

race_order  = ["white","black or african american","asian",
               "american indian or alaska native","not reported","unknown"]
race_labels = ["White","Black/AA","Asian","AIAN","Not reported","Unknown"]

ridh = df.groupby(["race","idh_status"]).size().unstack(fill_value=0)
pres = [r for r in race_order if r in ridh.index]
plab = [race_labels[race_order.index(r)] for r in pres]
wt_v = [ridh.loc[r,"wildtype"] if "wildtype" in ridh.columns else 0 for r in pres]
mt_v = [ridh.loc[r,"mutant"]   if "mutant"   in ridh.columns else 0 for r in pres]
x    = np.arange(len(pres))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.bar(x, wt_v, color=WT,  width=0.55, label="IDH-wildtype",
       zorder=3, edgecolor="white", lw=0.5)
ax.bar(x, mt_v, color=MUT, width=0.55, label="IDH-mutant",
       zorder=3, edgecolor="white", lw=0.5, bottom=wt_v)
ax.set_xticks(x); ax.set_xticklabels(plab, rotation=20, ha="right")
ax.set_ylabel("Cases"); ax.set_title("(a) Absolute counts by race")
ax.legend(frameon=False); ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
for i, (w, m) in enumerate(zip(wt_v, mt_v)):
    ax.text(i, w+m+1.5, str(w+m), ha="center", fontsize=7.5)

ax = axes[1]
tots = [w+m for w,m in zip(wt_v, mt_v)]
pcts = [100*m/t if t>0 else 0 for m,t in zip(mt_v, tots)]
bars = ax.bar(x, pcts,
              color=[MUT if p>=50 else WT for p in pcts],
              width=0.55, zorder=3, edgecolor="white", lw=0.5, alpha=0.85)
ax.axhline(50, color="#2C3E50", ls="--", lw=1, alpha=0.6, label="50%")
ax.set_xticks(x); ax.set_xticklabels(plab, rotation=20, ha="right")
ax.set_ylabel("% IDH-mutant")
ax.set_title(f"(b) IDH-mutant proportion  "
             f"(White vs Other: p={p_race:.3f})")
ax.set_ylim(0, 115)
ax.legend(frameon=False); ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
for bar, p, t in zip(bars, pcts, tots):
    ax.text(bar.get_x()+bar.get_width()/2, p+2,
            f"{p:.0f}%\n(n={t})",
            ha="center", va="bottom", fontsize=7.5)

plt.suptitle("Race × IDH Status  Fairness Analysis",
             fontsize=10, fontweight="bold", y=1.02)
plt.tight_layout()
save("fig7_race_idh_fairness")

# 
# FIG 8  Site-level IDH balance
# 
print("Fig 8  Site IDH balance")

ss = (
    df.groupby("site")["idh_status"]
    .value_counts().unstack(fill_value=0)
    .assign(total=lambda d: d.sum(axis=1))
    .query("total >= 10")
    .assign(pct=lambda d: 100*d.get("mutant",0)/d["total"])
    .sort_values("pct", ascending=False)
)

fig, ax = plt.subplots(figsize=(9, max(4, len(ss)*0.32)))
bars = ax.barh(range(len(ss)), ss["pct"],
               color=[MUT if p>=50 else WT for p in ss["pct"]],
               alpha=0.85, zorder=3, edgecolor="white", lw=0.4)
ax.axvline(50, color="#2C3E50", ls="--", lw=1, alpha=0.7,
           label="50%", zorder=4)
ax.set_yticks(range(len(ss))); ax.set_yticklabels(ss.index, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("% IDH-mutant")
ax.set_title("Site-Level IDH Balance  (sites n10)", fontsize=9)
ax.set_xlim(0, 118)
ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
ax.legend(frameon=False)
for bar, (_, row) in zip(bars, ss.iterrows()):
    ax.text(row["pct"]+1.5, bar.get_y()+bar.get_height()/2,
            f"{row['pct']:.0f}%  (n={int(row['total'])})",
            va="center", fontsize=6.5)
plt.tight_layout()
save("fig8_site_idh_balance")

# 
# FIG 9  Brain region × IDH status
# 
print("Fig 9  Brain region × IDH status")

region_counts = df["brain_region"].value_counts()
keep  = region_counts[region_counts >= 5].index
df_r  = df[df["brain_region"].isin(keep)].copy()

rs = (
    df_r.groupby("brain_region")["idh_status"]
    .value_counts().unstack(fill_value=0)
    .assign(total=lambda d: d.sum(axis=1))
    .assign(pct=lambda d: 100*d.get("mutant",0)/d["total"])
    .sort_values("pct", ascending=False)
)

fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, len(rs)*0.38)))
y     = np.arange(len(rs))
wt_r  = [rs.loc[r,"wildtype"] if "wildtype" in rs.columns else 0 for r in rs.index]
mut_r = [rs.loc[r,"mutant"]   if "mutant"   in rs.columns else 0 for r in rs.index]

ax = axes[0]
ax.barh(y, wt_r,  color=WT,  label="IDH-wildtype",
        zorder=3, edgecolor="white", lw=0.4)
ax.barh(y, mut_r, color=MUT, label="IDH-mutant",
        zorder=3, edgecolor="white", lw=0.4, left=wt_r)
ax.set_yticks(y); ax.set_yticklabels(rs.index, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Cases"); ax.set_title("(a) Absolute counts")
ax.legend(frameon=False); ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
for i, (w, m) in enumerate(zip(wt_r, mut_r)):
    ax.text(w+m+0.3, i, str(w+m), va="center", fontsize=7.5)

ax = axes[1]
bars = ax.barh(y, rs["pct"],
               color=[MUT if p>=50 else WT for p in rs["pct"]],
               alpha=0.85, zorder=3, edgecolor="white", lw=0.4)
ax.axvline(50, color="#2C3E50", ls="--", lw=1, alpha=0.7, label="50%")
ax.set_yticks(y); ax.set_yticklabels(rs.index, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("% IDH-mutant"); ax.set_title("(b) IDH-mutant proportion")
ax.set_xlim(0, 115)
ax.legend(frameon=False); ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
for bar, (_, row) in zip(bars, rs.iterrows()):
    ax.text(row["pct"]+1.5, bar.get_y()+bar.get_height()/2,
            f"{row['pct']:.0f}%  (n={int(row['total'])})",
            va="center", fontsize=7.5)

plt.suptitle("Brain Region × IDH Status",
             fontsize=10, fontweight="bold", y=1.02)
plt.tight_layout()
save("fig9_brain_region_idh")

print("\nGenerating summary CSV...")

rows = []

def make_row(stratum, group, sub):
    n       = len(sub)
    n_mut   = (sub["idh_status"] == "mutant").sum()
    n_wt    = (sub["idh_status"] == "wildtype").sum()
    ages    = sub["age"].dropna()
    return {
        "stratum":          stratum,
        "group":            group,
        "n":                n,
        "n_idh_mutant":     n_mut,
        "n_idh_wildtype":   n_wt,
        "pct_idh_mutant":   f"{100*n_mut/n:.1f}%" if n > 0 else "?",
        "pct_idh_wildtype":  f"{100*n_wt/n:.1f}%"  if n > 0 else "?",
        "age_median":       f"{ages.median():.0f}" if len(ages) > 0 else "?",
        "age_q1":           f"{ages.quantile(0.25):.0f}" if len(ages) > 0 else "?",
        "age_q3":           f"{ages.quantile(0.75):.0f}" if len(ages) > 0 else "?",
        "pct_male":         f"{100*(sub['sex']=='male').mean():.1f}%",
    }

# overall
rows.append(make_row("Overall", "All usable cases", df))

# by project
for proj in ["TCGA-GBM", "TCGA-LGG"]:
    rows.append(make_row("Project", proj, df[df["project"] == proj]))

# by race
race_order_full = [
    ("white",                            "White"),
    ("black or african american",        "Black/AA"),
    ("asian",                            "Asian"),
    ("american indian or alaska native", "AIAN"),
    ("not reported",                     "Not reported"),
    ("unknown",                          "Unknown"),
]
for race_raw, race_label in race_order_full:
    sub = df[df["race"] == race_raw]
    if len(sub) > 0:
        rows.append(make_row("Race", race_label, sub))

# by sex
for sex_val, sex_label in [("male","Male"),("female","Female"),
                            ("not reported","Not reported")]:
    sub = df[df["sex"] == sex_val]
    if len(sub) > 0:
        rows.append(make_row("Sex", sex_label, sub))

# by site (all sites with n >= 1)
for site in df["site"].value_counts().index:
    sub = df[df["site"] == site]
    rows.append(make_row("Site", site, sub))

# by brain region
for region in df["brain_region"].value_counts().index:
    sub = df[df["brain_region"] == region]
    rows.append(make_row("Brain region", region, sub))

df_summary = pd.DataFrame(rows)
df_summary.to_csv(OUT / "summary_table.csv", index=False)
print(f"  ? summary_table.csv  ({len(df_summary)} rows)")
print(df_summary.head(10).to_string(index=False))

print(f"\nDone  all figures saved to {OUT}/")