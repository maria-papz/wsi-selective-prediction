# scripts/build_r0_figures.py
"""
R0 (participant flow + cohort overview) figures for the Results chapter.
Recomputes every number from source data on each run -- nothing here is
hardcoded, so the figures stay correct if a cohort's curation changes again
(as already happened once with the CALIBRATION_SITES Fondazione-Besta fix).

Input:
    data/raw/ebrains/clinical/annotation.csv
    data/raw/tcga/clinical/{tcga_demographics.tsv, tcga_slides_clean.csv,
                            tcga_idh_final.csv, tcga_manifest.csv}
    data/raw/ipd_brain/clinical/ipd_brain_v1.csv
    outputs/ipd_brain_curation/patient_level_curation.csv

Output:
    outputs/r0_figures/figure1_participant_flow.png
    outputs/r0_figures/figure1b_cohort_overview_3panel.png
"""

import re
import warnings
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")
pd.set_option("future.no_silent_downcasting", True)

OUT = Path("outputs/r0_figures")
OUT.mkdir(parents=True, exist_ok=True)


from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend("pgf", FigureCanvasPgf)
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,  # defer to the thesis document's own font, don't embed one
})

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

# UCL brand palette (Lead colours: dark purple, white, bright purple, mid purple;
# supporting: light/pale purple, heritage blue).
DARK_PURPLE = "#361a54"
BRIGHT_PURPLE = "#993bff"
MID_PURPLE = "#ba82ff"
LIGHT_PURPLE = "#ddbdff"
PALE_PURPLE = "#eedeff"
HERITAGE_BLUE = "#30d6ff"
OFF_WHITE = "#fafafa"

WT, MUT = DARK_PURPLE, BRIGHT_PURPLE
MALE, FEMALE = DARK_PURPLE, HERITAGE_BLUE
GREY = "#4a4a4a"  # neutral, for arrows/annotation text only -- not part of the brand palette


# =============================================================================
# 1. EBRAINS funnel + crosstabs
# =============================================================================
def load_ebrains():
    df_all = pd.read_csv("data/raw/ebrains/clinical/annotation.csv")
    n_raw_patients = df_all["pat_id"].nunique()
    n_raw_slides = len(df_all)

    def extract_idh(d):
        d = str(d).lower()
        if "idh-wildtype" in d or "idh-wild" in d:
            return "wildtype"
        if "idh-mutant" in d:
            return "mutant"
        return None

    def extract_type(d):
        d = str(d).lower()
        if "glioblastoma" in d:
            return "GBM"
        if "oligodendroglioma" in d:
            return "Oligo"
        if "astrocytoma" in d:
            return "Astro"
        return "other"

    df_all["idh_status"] = df_all["diagnosis"].apply(extract_idh)
    df_idh = df_all[df_all["idh_status"].notna()].copy()
    n_idh_patients = df_idh["pat_id"].nunique()
    n_idh_slides = len(df_idh)

    patients_primary = df_idh[df_idh["recurrence"] == 0]["pat_id"].unique()
    n_excl_recurrence = n_idh_patients - len(patients_primary)

    df_p = df_idh[df_idh["pat_id"].isin(patients_primary) & (df_idh["recurrence"] == 0)]
    n_pre_dedup_slides = len(df_p)

    df_dedup = (
        df_p.sort_values("tissue_area", ascending=False)
        .drop_duplicates(subset="pat_id", keep="first")
        .copy()
    )
    df_dedup["tumour_type"] = df_dedup["diagnosis"].apply(extract_type)

    funnel = dict(
        n_raw_patients=n_raw_patients, n_raw_slides=n_raw_slides,
        n_excl_no_idh=n_raw_patients - n_idh_patients,
        n_idh_patients=n_idh_patients, n_idh_slides=n_idh_slides,
        n_excl_recurrence=n_excl_recurrence,
        n_post_recurrence_patients=len(patients_primary),
        n_post_recurrence_slides=n_pre_dedup_slides,
        n_final=len(df_dedup),
    )
    return df_dedup, funnel



CALIBRATION_SITES = {
    "University of Florida",
    "Milan - Italy, Fondazione IRCCS Instituto Neuroligico C. Besta",
    "Fondazione-Besta",
    "Mayo Clinic - Rochester",
}


def load_tcga():
    df_demo = pd.read_csv("data/raw/tcga/clinical/tcga_demographics.tsv", sep="\t")
    df_demo["case_id"] = df_demo["submitter_id"].str[:12]
    n_raw = len(df_demo)

    df_slides = pd.read_csv("data/raw/tcga/clinical/tcga_slides_clean.csv")
    df_dx = df_slides[df_slides["is_dx"]].copy()
    df_dx_dedup = (
        df_dx.sort_values("file_size", ascending=False)
        .drop_duplicates(subset="case_id", keep="first")
        .copy()
    )
    df_dx_dedup["size_mb"] = df_dx_dedup["file_size"] / 1e6
    df_dx_dedup["size_flag"] = (df_dx_dedup["size_mb"] < 50).astype(bool)

    df_idh = pd.read_csv("data/raw/tcga/clinical/tcga_idh_final.csv")

    df_merged = (
        df_demo
        .merge(df_idh[["case_id", "idh_status"]], on="case_id", how="left")
        .merge(df_dx_dedup[["case_id", "size_mb", "size_flag"]], on="case_id", how="left")
    )

    pool = df_merged.copy()
    n_pool = len(pool)

    pool1 = pool[pool["size_mb"].notna()].copy()
    n_excl_no_dx = n_pool - len(pool1)

    pool1["size_flag"] = pool1["size_flag"].fillna(False).astype(bool)
    pool2 = pool1[~pool1["size_flag"]].copy()
    n_excl_small = len(pool1) - len(pool2)

    pool3 = pool2[pool2["idh_status"].notna()].copy()
    n_excl_no_idh = len(pool2) - len(pool3)

    # final manifest carries site/race/sex for the split + Table 1
    manifest = pd.read_csv("data/raw/tcga/clinical/tcga_manifest.csv")
    assert len(manifest) == len(pool3), (
        f"Reconstructed funnel final n ({len(pool3)}) doesn't match "
        f"tcga_manifest.csv ({len(manifest)}) -- inputs have drifted, check "
        f"fetch_tcga_metadata.py was rerun consistently."
    )

    is_calib = manifest["site"].isin(CALIBRATION_SITES)
    calib, evalset = manifest[is_calib], manifest[~is_calib]

    who = pd.read_csv("data/raw/tcga/clinical/Matrix_WHO2021.csv")
    who["case_id"] = who["Patient_ID"].str[:12]
    manifest_who = manifest.merge(
        who[["case_id", "classification.2021_simplified.labels"]],
        on="case_id", how="left",
    )

    funnel = dict(
        n_raw=n_pool, n_excl_no_dx=n_excl_no_dx, n_after_dx=len(pool1),
        n_excl_small=n_excl_small, n_after_small=len(pool2),
        n_excl_no_idh=n_excl_no_idh, n_final=len(manifest),
        n_calib=len(calib), n_calib_mutant=(calib["idh_status"] == "mutant").sum(),
        n_calib_wt=(calib["idh_status"] == "wildtype").sum(),
        n_calib_sites=calib["site"].nunique(),
        n_eval=len(evalset), n_eval_mutant=(evalset["idh_status"] == "mutant").sum(),
        n_eval_wt=(evalset["idh_status"] == "wildtype").sum(),
        n_eval_sites=evalset["site"].nunique(),
    )
    return manifest, manifest_who, calib, evalset, funnel


# =============================================================================
# 3. IPD Brain funnel + crosstabs
# =============================================================================
SLIDE_SUFFIX_RE = re.compile(r"\([a-z]\)$", re.IGNORECASE)


def load_ipd_brain():
    raw = pd.read_csv("data/raw/ipd_brain/clinical/ipd_brain_v1.csv", encoding="utf-8-sig")
    n_raw = len(raw)

    def pid(row):
        sids = [s.strip() for s in str(row["Case Number"]).split("\n") if s.strip()]
        return SLIDE_SUFFIX_RE.sub("", sids[0]).strip()

    raw["patient_id"] = raw.apply(pid, axis=1)

    pl = pd.read_csv("outputs/ipd_brain_curation/patient_level_curation.csv")
    n_excl_reason = pl.loc[pl.status == "excluded", "reason"].value_counts()

    included = pl[pl.status == "included"].copy()

    def extract_type(d):
        d = str(d).lower()
        if "glioblastoma" in d:
            return "GBM"
        if "oligodendroglioma" in d:
            return "Oligo"
        if "astrocytoma" in d:
            return "Astro"
        return "other"

    included["subtype"] = included["diagnosis"].apply(extract_type)
    included["idh_status"] = included["idh1r132h"].map({1: "mutant", 0: "wildtype"})

    merged = included.merge(
        raw[["patient_id", "Sex", "WHO Grade"]].drop_duplicates("patient_id"),
        on="patient_id", how="left",
    )

    funnel = dict(
        n_raw=n_raw,
        n_excl_recurrence=n_excl_reason.get(
            "post-recurrence-only presentation (no primary slide in cohort)", 0
        ),
        n_excl_unprocessable=n_excl_reason.get(
            "all listed slides unprocessable (see per-slide reason)", 0
        ),
        n_final=len(included),
        n_mutant=(included["idh_status"] == "mutant").sum(),
        n_wt=(included["idh_status"] == "wildtype").sum(),
    )
    return merged, funnel


# =============================================================================
# Plot helpers
# =============================================================================
def stacked_idh_bar(ax, cats, wt_n, mut_n, title, legend=False):
    x = np.arange(len(cats))
    ax.bar(x, wt_n, width=0.55, color=WT, label="IDH-wildtype", zorder=3)
    ax.bar(x, mut_n, width=0.55, color=MUT, label="IDH-mutant", bottom=wt_n, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Cases")
    ax.set_title(title)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    if legend:
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    for i, (w, m) in enumerate(zip(wt_n, mut_n)):
        t = w + m
        ax.text(i, t + t * 0.02, str(t), ha="center", fontsize=8, fontweight="bold")
        if w:
            ax.text(i, w / 2, str(w), ha="center", va="center", color="white",
                     fontsize=7, fontweight="bold")
        if m:
            ax.text(i, w + m / 2, str(m), ha="center", va="center", color="white",
                     fontsize=7, fontweight="bold")


def sex_barh(ax, male_n, female_n, title, other_n=0, other_label="not recorded"):
    cats, vals, colors = ["male", "female"], [male_n, female_n], [MALE, FEMALE]
    if other_n:
        cats.append(other_label)
        vals.append(other_n)
        colors.append(GREY)
    total = sum(vals)
    bars = ax.barh(cats, vals, color=colors, edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlabel("Cases")
    ax.set_title(title)
    ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
    for bar, v in zip(bars, vals):
        ax.text(v + total * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v} ({100 * v / total:.0f}%)", va="center", fontsize=7.5)


def box(ax, x, y, w, h, text, fc=None, ec=None, fontsize=11, fontweight="normal"):
    # Same box geometry/positions as the original diagram -- per instruction,
    # only the fontsize changed (supervisor: "text is so small"), nothing
    # about layout, columns, or content.
    fc = PALE_PURPLE if fc is None else fc
    ec = DARK_PURPLE if ec is None else ec
    r = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
        fc=fc, ec=ec, lw=1.1, zorder=3,
    )
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
             fontweight=fontweight, zorder=4, linespacing=1.3)


def arrow_down(ax, x, y_top, y_bot, label=None):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.1), zorder=2)
    if label:
        ax.text(x + 0.15, (y_top + y_bot) / 2, label, fontsize=9.5, va="center",
                 ha="left", color="#555555")


# =============================================================================
# Main
# =============================================================================
def main():
    ebrains_df, ebrains_f = load_ebrains()
    tcga_manifest, tcga_who, tcga_calib, tcga_eval, tcga_f = load_tcga()
    ipd_df, ipd_f = load_ipd_brain()

    print("EBRAINS funnel:", ebrains_f)
    print("TCGA funnel:", tcga_f)
    print("IPD Brain funnel:", ipd_f)

    # ---------------- Figure 1b: 3-cohort overview panel ----------------
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5))
    fig.suptitle(
        "Cohort overview: EBRAINS (training), TCGA (external), IPD Brain (external)",
        fontsize=12, fontweight="bold", y=0.995,
    )

    # Row 1: EBRAINS
    ct = ebrains_df.groupby(["tumour_type", "idh_status"]).size().unstack(fill_value=0)
    ct = ct.reindex(["GBM", "Astro", "Oligo"]).fillna(0).astype(int)
    stacked_idh_bar(axes[0, 0], ct.index.tolist(), ct["wildtype"].tolist(),
                     ct["mutant"].tolist(),
                     f"(a) EBRAINS: IDH by tumour type (n={ebrains_f['n_final']})",
                     legend=True)
    sex_ct = ebrains_df["sex"].value_counts(dropna=False)
    n_missing = ebrains_df["sex"].isna().sum()
    sex_barh(axes[0, 1], sex_ct.get("male", 0), sex_ct.get("female", 0),
             "(b) EBRAINS: Sex", other_n=n_missing, other_label="not recorded")
    ax = axes[0, 2]
    ax.axis("off")
    ax.text(0.5, 0.5, "Race not recorded\n(EBRAINS)", ha="center", va="center",
             fontsize=9, color=GREY, transform=ax.transAxes, style="italic")
    ax.set_title("(c) EBRAINS: Race")

    # Row 2: TCGA
    proj_ct = tcga_manifest.groupby(["project", "idh_status"]).size().unstack(fill_value=0)
    proj_ct = proj_ct.reindex(["TCGA-GBM", "TCGA-LGG"]).fillna(0).astype(int)
    # Labelled "TCGA-GBM/LGG (project)", not "GBM/LGG", deliberately: this is GDC's
    # legacy tissue-collection-era project categorisation, not the WHO-2021
    # reclassified subtype (that's Table 1, sourced from Mendonca et al. 2025's
    # matrix, where "glioblastoma" is 0/339 mutant by definition). TCGA-GBM
    # legitimately contains a small IDH-mutant fraction under the older scheme --
    # conflating the two labels is exactly the ambiguity this caption avoids.
    stacked_idh_bar(axes[1, 0], ["TCGA-GBM", "TCGA-LGG"], proj_ct["wildtype"].tolist(),
                     proj_ct["mutant"].tolist(),
                     f"(d) TCGA: IDH by project (n={tcga_f['n_final']})")
    sex_ct = tcga_manifest["sex"].str.lower().value_counts(dropna=False)
    sex_barh(axes[1, 1], sex_ct.get("male", 0), sex_ct.get("female", 0), "(e) TCGA: Sex",
             other_n=sex_ct.get("not reported", 0), other_label="not reported")
    ax = axes[1, 2]
    race_map = {
        "white": "White", "black or african american": "Black/AA", "asian": "Asian",
        "american indian or alaska native": "AIAN", "not reported": "Not rep.",
    }
    race_ct = tcga_manifest["race"].str.lower().map(lambda x: race_map.get(x, "Unknown")).value_counts()
    order = ["White", "Black/AA", "Asian", "AIAN", "Not rep.", "Unknown"]
    race_vals = [race_ct.get(o, 0) for o in order]
    bars = ax.barh(order[::-1], race_vals[::-1], color=DARK_PURPLE, edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlabel("Cases")
    ax.set_title("(f) TCGA: Race")
    ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
    for bar, v in zip(bars, race_vals[::-1]):
        ax.text(v + max(race_vals) * 0.02, bar.get_y() + bar.get_height() / 2, str(v),
                 va="center", fontsize=7.5)

    # Row 3: IPD Brain
    ct = ipd_df.groupby(["subtype", "idh_status"]).size().unstack(fill_value=0)
    ct = ct.reindex(["GBM", "Astro", "Oligo"]).fillna(0).astype(int)
    stacked_idh_bar(axes[2, 0], ct.index.tolist(), ct["wildtype"].tolist(),
                     ct["mutant"].tolist(), f"(g) IPD Brain: IDH by subtype (n={ipd_f['n_final']})")
    sex_ct = ipd_df["Sex"].str.upper().value_counts(dropna=False)
    n_other = len(ipd_df) - sex_ct.get("M", 0) - sex_ct.get("F", 0)
    sex_barh(axes[2, 1], sex_ct.get("M", 0), sex_ct.get("F", 0), "(h) IPD Brain: Sex",
             other_n=n_other, other_label="not recorded")
    ax = axes[2, 2]
    grade_ct = ipd_df["WHO Grade"].value_counts().sort_index()
    bars = ax.bar(grade_ct.index.astype(str), grade_ct.values, color=MID_PURPLE,
                   edgecolor="white", lw=0.5, zorder=3)
    ax.set_ylabel("Cases")
    ax.set_xlabel("WHO Grade")
    ax.set_title("(i) IPD Brain: WHO Grade")
    ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    for bar, v in zip(bars, grade_ct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(grade_ct.values) * 0.02,
                 str(v), ha="center", fontsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT / "figure1b_cohort_overview_3panel.png")
    plt.savefig(OUT / "figure1b_cohort_overview_3panel.pdf")  # vector, for \includegraphics
    plt.close()
    print("Saved figure1b_cohort_overview_3panel.{png,pdf}")

    # ---------------- Figure 1: participant flow diagram ----------------
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(3.2, 11.6)  # tight to actual content: lowest box (calib/eval split) sits at y=3.55
    ax.axis("off")

    col_x, col_w = [0.3, 4.6, 8.9], 3.7
    titles = ["EBRAINS (training)", "TCGA (external)", "IPD Brain (external)"]
    for cx, t in zip(col_x, titles):
        ax.text(cx + col_w / 2, 11.15, t, ha="center", fontsize=12, fontweight="bold")

    # EBRAINS column
    x = col_x[0]
    box(ax, x, 9.9, col_w, 1.0,
        f"{ebrains_f['n_raw_patients']:,} patients\n({ebrains_f['n_raw_slides']:,} slides, EBRAINS Atlas)")
    arrow_down(ax, x + col_w / 2, 9.9, 9.15,
               f"excl. {ebrains_f['n_excl_no_idh']:,}:\nno confirmed IDH status")
    box(ax, x, 8.4, col_w, 0.75,
        f"{ebrains_f['n_idh_patients']} patients ({ebrains_f['n_idh_slides']} slides)")
    arrow_down(ax, x + col_w / 2, 8.4, 7.65,
               f"excl. {ebrains_f['n_excl_recurrence']}:\npost-recurrence-only")
    box(ax, x, 6.9, col_w, 0.75,
        f"{ebrains_f['n_post_recurrence_patients']} patients ({ebrains_f['n_post_recurrence_slides']} slides)")
    n_dedup_slides = ebrains_f["n_post_recurrence_slides"] - ebrains_f["n_final"]
    arrow_down(ax, x + col_w / 2, 6.9, 6.15,
               f"dedup: largest tissue\narea kept ({n_dedup_slides} slides, 0 pts)")
    n_wt = (ebrains_df["idh_status"] == "wildtype").sum()
    n_mut = (ebrains_df["idh_status"] == "mutant").sum()
    box(ax, x, 5.4, col_w, 0.75,
        f"FINAL n={ebrains_f['n_final']}\n{n_wt} wildtype / {n_mut} mutant",
        fc=LIGHT_PURPLE, ec=BRIGHT_PURPLE, fontweight="bold")

    # TCGA column
    x = col_x[1]
    box(ax, x, 9.9, col_w, 1.0, f"{tcga_f['n_raw']:,} GDC cases identified")
    arrow_down(ax, x + col_w / 2, 9.9, 9.15,
               f"excl. {tcga_f['n_excl_no_dx']}:\nno diagnostic (DX) slide")
    box(ax, x, 8.4, col_w, 0.75, f"{tcga_f['n_after_dx']} cases")
    arrow_down(ax, x + col_w / 2, 8.4, 7.65,
               f"excl. {tcga_f['n_excl_small']}:\nDX slide < 50MB")
    box(ax, x, 6.9, col_w, 0.75, f"{tcga_f['n_after_small']} cases")
    arrow_down(ax, x + col_w / 2, 6.9, 6.15,
               f"excl. {tcga_f['n_excl_no_idh']}: no molecular\nIDH determination")
    n_mut = (tcga_manifest["idh_status"] == "mutant").sum()
    n_wt = (tcga_manifest["idh_status"] == "wildtype").sum()
    box(ax, x, 5.4, col_w, 0.75, f"FINAL n={tcga_f['n_final']}\n{n_mut} mutant / {n_wt} wildtype",
        fc=LIGHT_PURPLE, ec=BRIGHT_PURPLE, fontweight="bold")
    arrow_down(ax, x + col_w / 2, 5.4, 4.65, "split by site")
    box(ax, x, 3.55, col_w * 0.47, 1.0,
        f"Calibration\nn={tcga_f['n_calib']} ({tcga_f['n_calib_sites']} sites)\n"
        f"{tcga_f['n_calib_mutant']} mutant/{tcga_f['n_calib_wt']} WT",
        fc=PALE_PURPLE, ec=MID_PURPLE, fontsize=9)
    box(ax, x + col_w * 0.53, 3.55, col_w * 0.47, 1.0,
        f"Evaluation\nn={tcga_f['n_eval']} ({tcga_f['n_eval_sites']} sites)\n"
        f"{tcga_f['n_eval_mutant']} mutant/{tcga_f['n_eval_wt']} WT",
        fc=PALE_PURPLE, ec=MID_PURPLE, fontsize=9)

    # IPD Brain column
    x = col_x[2]
    box(ax, x, 9.9, col_w, 1.0, f"{ipd_f['n_raw']} patient records\n(Chauhan et al. 2024)")
    arrow_down(ax, x + col_w / 2, 9.9, 9.15,
               f"excl. {ipd_f['n_excl_recurrence']}: post-recurrence-only\n"
               f"excl. {ipd_f['n_excl_unprocessable']}: unprocessable WSI")
    box(ax, x, 8.4, col_w, 0.75,
        f"FINAL n={ipd_f['n_final']}\n{ipd_f['n_mutant']} mutant / {ipd_f['n_wt']} wildtype",
        fc=LIGHT_PURPLE, ec=BRIGHT_PURPLE, fontweight="bold")

    plt.savefig(OUT / "figure1_participant_flow.png")  # quick preview only
    plt.savefig(OUT / "figure1_participant_flow.pdf")   # vector fallback, \includegraphics
    fig.canvas.switch_backends(FigureCanvasPgf)
    plt.savefig(OUT / "figure1_participant_flow.pgf")   # \input{} into the thesis, native font
    plt.close()
    print("Saved figure1_participant_flow.{png,pdf,pgf}")


if __name__ == "__main__":
    main()
