import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ast
import os

# -- output dirs --------------------------------------------------------------
os.makedirs("data/raw/ebrains/clinical", exist_ok=True)
os.makedirs("outputs/ebrains_eda", exist_ok=True)

PALETTE = {"mutant": "#2196F3", "wildtype": "#F44336"}
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# -- load ---------------------------------------------------------------------
df_all = pd.read_csv("data/raw/ebrains/clinical/annotation.csv")

print("=" * 60)
print("RAW DATA")
print("=" * 60)
print(f"Total rows:       {len(df_all)}")
print(f"Unique patients:  {df_all['pat_id'].nunique()}")
print(f"Columns:          {df_all.columns.tolist()}\n")

# -- helpers -------------------------------------------------------------------
def extract_idh(diagnosis):
    d = str(diagnosis).lower()
    if "idh-wildtype" in d or "idh-wild" in d:
        return "wildtype"
    elif "idh-mutant" in d:
        return "mutant"
    return None

def extract_tumour_type(diagnosis):
    d = str(diagnosis).lower()
    if "glioblastoma" in d:
        return "GBM"
    elif "oligodendroglioma" in d:
        return "oligodendroglioma"
    elif "astrocytoma" in d:
        return "astrocytoma"
    return "other"

def parse_location(loc):
    try:
        locs = ast.literal_eval(loc)
        return ", ".join(locs) if locs else "not specified"
    except Exception:
        return "not specified"

CEREBRAL = [
    "frontal", "temporal", "parietal", "occipital",
    "insular", "cerebral", "basal ganglia", "diencephalon"
]

def is_cerebral(loc):
    return any(kw in str(loc).lower() for kw in CEREBRAL)

# -- IDH test ------------------------------------------------------------------
print("=" * 60)
print("IDH EXTRACTION TEST")
print("=" * 60)
tests = [
    "Glioblastoma, IDH-wildtype",
    "Anaplastic oligodendroglioma, IDH-mutant and 1p/19q codeleted",
    "Gemistocytic astrocytoma, IDH-mutant",
    "Diffuse astrocytoma, IDH-wildtype",
    "Anaplastic astrocytoma, IDH-mutant",
    "Glioblastoma, IDH-mutant",
    "Pilocytic astrocytoma",
    "Meningothelial meningioma",
]
for t in tests:
    print(f"  {t[:52]:<52} - {extract_idh(t)}")

# -- filter to IDH-annotated ---------------------------------------------------
df_all["idh_status"] = df_all["diagnosis"].apply(extract_idh)
df_idh = df_all[df_all["idh_status"].notna()].copy()

print(f"\n{'='*60}")
print("IDH-ANNOTATED ROWS (before dedup)")
print("=" * 60)
print(f"Total rows:       {len(df_idh)}")
print(f"Unique patients:  {df_idh['pat_id'].nunique()}")
print("\nIDH distribution:\n", df_idh["idh_status"].value_counts().to_string())

# -- remove patients with ONLY recurrence slides -------------------------------
patients_with_primary = df_idh[df_idh["recurrence"] == 0]["pat_id"].unique()
df_idh_primary_only = df_idh[df_idh["pat_id"].isin(patients_with_primary)].copy()
removed = df_idh["pat_id"].nunique() - len(patients_with_primary)

print(f"\n{'='*60}")
print("RECURRENCE FILTER")
print("=" * 60)
print(f"Patients with at least one primary slide: {len(patients_with_primary)}")
print(f"Patients removed (recurrence-only):       {removed}")

# -- deduplicate: keep primary slide only (recurrence=0) ----------------------
df_deduped = (
    df_idh_primary_only[df_idh_primary_only["recurrence"] == 0]
    .sort_values("tissue_area", ascending=False)   # largest tissue area as tiebreak
    .drop_duplicates(subset="pat_id", keep="first")
    .copy()
)

# -- enrich --------------------------------------------------------------------
df_deduped["tumour_type"]    = df_deduped["diagnosis"].apply(extract_tumour_type)
df_deduped["location_parsed"]= df_deduped["location"].apply(parse_location)
df_deduped["is_cerebral"]    = df_deduped["location_parsed"].apply(is_cerebral)
df_deduped["idh_binary"]     = (df_deduped["idh_status"] == "mutant").astype(int)
df_deduped["dataset"]        = "ebrains"
df_deduped["split"]          = "train"

print(f"\n{'='*60}")
print("FINAL COHORT AFTER DEDUP")
print("=" * 60)
print(f"Cases:            {len(df_deduped)}")
print(f"\nIDH status:\n{df_deduped['idh_status'].value_counts().to_string()}")
print(f"\nTumour type:\n{df_deduped['tumour_type'].value_counts().to_string()}")
print(f"\nSex:\n{df_deduped['sex'].value_counts().to_string()}")
print(f"\nAge:\n{df_deduped['age'].describe().to_string()}")
print(f"\nCerebral: {df_deduped['is_cerebral'].sum()} / Not: {(~df_deduped['is_cerebral']).sum()}")
print(f"\nMissing data:")
for col in ["age", "sex", "location", "idh_status", "grade"]:
    print(f"  {col:<20} missing: {df_deduped[col].isna().sum()}")

# -- save CSV ------------------------------------------------------------------
out_csv = "data/raw/ebrains/clinical/ebrains_idh_cases.csv"
df_deduped.to_csv(out_csv, index=False)
print(f"\nSaved {len(df_deduped)} cases to {out_csv}")

# save UUID list for download script
uuids = df_deduped["uuid"].tolist()
with open("manifests/ebrains_uuids.txt", "w") as f:
    for u in uuids:
        f.write(u + "\n")
print(f"Saved {len(uuids)} UUIDs to manifests/ebrains_uuids.txt")

# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------

# -- 1. IDH class balance ------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
counts = df_deduped["idh_status"].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[PALETTE[k] for k in counts.index], edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha="center", va="bottom", fontweight="bold")
ax.set_title("IDH Status Distribution (Training Set)", fontweight="bold")
ax.set_ylabel("Number of cases")
ax.set_xlabel("")
ax.set_ylim(0, counts.max() * 1.15)
plt.tight_layout()
plt.savefig("outputs/ebrains_eda/01_idh_class_balance.png")
plt.close()

# -- 2. IDH status by tumour type ---------------------------------------------
ct = df_deduped.groupby(["tumour_type", "idh_status"]).size().unstack(fill_value=0)
ct = ct.reindex(["GBM", "astrocytoma", "oligodendroglioma"])
fig, ax = plt.subplots(figsize=(7, 4))
ct.plot(kind="bar", ax=ax, color=[PALETTE["mutant"], PALETTE["wildtype"]],
        edgecolor="white", width=0.6)
ax.set_title("IDH Status by Tumour Type", fontweight="bold")
ax.set_ylabel("Number of cases")
ax.set_xlabel("")
ax.set_xticklabels(ct.index, rotation=0)
ax.legend(title="IDH status")
for container in ax.containers:
    ax.bar_label(container, padding=2)
plt.tight_layout()
plt.savefig("outputs/ebrains_eda/02_idh_by_tumour_type.png")
plt.close()

# -- 3. Age distribution by IDH status ----------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for label, grp in df_deduped.groupby("idh_status"):
    grp["age"].dropna().plot.kde(ax=ax, label=label, color=PALETTE[label], linewidth=2)
    ax.axvline(grp["age"].median(), color=PALETTE[label],
               linestyle="--", linewidth=1, alpha=0.7)
ax.set_title("Age Distribution by IDH Status", fontweight="bold")
ax.set_xlabel("Age at diagnosis (years)")
ax.set_ylabel("Density")
ax.legend(title="IDH status")
plt.tight_layout()
plt.savefig("outputs/ebrains_eda/03_age_by_idh.png")
plt.close()

# -- 4. Sex by IDH status -----------------------------------------------------
sex_ct = df_deduped[df_deduped["sex"].isin(["male", "female"])]\
    .groupby(["sex", "idh_status"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(5, 4))
sex_ct.plot(kind="bar", ax=ax, color=[PALETTE["mutant"], PALETTE["wildtype"]],
            edgecolor="white", width=0.5)
ax.set_title("Sex by IDH Status", fontweight="bold")
ax.set_ylabel("Number of cases")
ax.set_xlabel("")
ax.set_xticklabels(sex_ct.index, rotation=0)
ax.legend(title="IDH status")
for container in ax.containers:
    ax.bar_label(container, padding=2)
plt.tight_layout()
plt.savefig("outputs/ebrains_eda/04_sex_by_idh.png")
plt.close()

# -- 5. Top locations by IDH status -------------------------------------------
loc_ct = df_deduped.groupby(["location_parsed", "idh_status"])\
    .size().unstack(fill_value=0)
loc_ct["total"] = loc_ct.sum(axis=1)
top_locs = loc_ct[loc_ct["total"] >= 5].sort_values("total", ascending=False)\
    .drop(columns="total").head(15)

fig, ax = plt.subplots(figsize=(10, 5))
top_locs.plot(kind="bar", ax=ax, color=[PALETTE["mutant"], PALETTE["wildtype"]],
              edgecolor="white", width=0.7)
ax.set_title("Top Tumour Locations by IDH Status\n(locations with -5 cases)", fontweight="bold")
ax.set_ylabel("Number of cases")
ax.set_xlabel("Location")
ax.set_xticklabels(top_locs.index, rotation=45, ha="right")
ax.legend(title="IDH status")
plt.tight_layout()
plt.savefig("outputs/ebrains_eda/05_location_by_idh.png")
plt.close()

# -- 6. Cerebral vs non-cerebral by IDH status --------------------------------
df_deduped["region"] = df_deduped["is_cerebral"].map(
    {True: "cerebral", False: "non-cerebral / unknown"})
reg_ct = df_deduped.groupby(["region", "idh_status"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(6, 4))
reg_ct.plot(kind="bar", ax=ax, color=[PALETTE["mutant"], PALETTE["wildtype"]],
            edgecolor="white", width=0.5)
ax.set_title("Cerebral vs Non-cerebral by IDH Status", fontweight="bold")
ax.set_ylabel("Number of cases")
ax.set_xlabel("")
ax.set_xticklabels(reg_ct.index, rotation=0)
ax.legend(title="IDH status")
for container in ax.containers:
    ax.bar_label(container, padding=2)
plt.tight_layout()
plt.savefig("outputs/ebrains_eda/06_cerebral_by_idh.png")
plt.close()

# -- 7. Grade distribution by IDH status --------------------------------------
grade_ct = df_deduped[df_deduped["grade"].notna()]\
    .groupby(["grade", "idh_status"]).size().unstack(fill_value=0)
grade_order = ["I", "II", "III", "IV"]
grade_ct = grade_ct.reindex([g for g in grade_order if g in grade_ct.index])
fig, ax = plt.subplots(figsize=(6, 4))
grade_ct.plot(kind="bar", ax=ax, color=[PALETTE["mutant"], PALETTE["wildtype"]],
              edgecolor="white", width=0.5)
ax.set_title("WHO Grade by IDH Status", fontweight="bold")
ax.set_ylabel("Number of cases")
ax.set_xlabel("WHO Grade")
ax.set_xticklabels(grade_ct.index, rotation=0)
ax.legend(title="IDH status")
for container in ax.containers:
    ax.bar_label(container, padding=2)
plt.tight_layout()
plt.savefig("outputs/ebrains_eda/07_grade_by_idh.png")
plt.close()

# -- 8. Age boxplot by tumour type and IDH ------------------------------------
df_plot = df_deduped[df_deduped["tumour_type"] != "other"].copy()
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df_plot, x="tumour_type", y="age", hue="idh_status",
            palette=PALETTE, ax=ax, width=0.5, flierprops={"markersize": 3})
ax.set_title("Age by Tumour Type and IDH Status", fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Age at diagnosis (years)")
ax.legend(title="IDH status")
plt.tight_layout()
plt.savefig("outputs/ebrains_eda/08_age_by_tumourtype_idh.png")
plt.close()

# -- 9. Tissue area distribution by IDH ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for label, grp in df_deduped.groupby("idh_status"):
    grp["tissue_area"].dropna().plot.kde(ax=ax, label=label,
                                          color=PALETTE[label], linewidth=2)
ax.set_title("Tissue Area Distribution by IDH Status", fontweight="bold")
ax.set_xlabel("Tissue area (mm²)")
ax.set_ylabel("Density")
ax.legend(title="IDH status")
plt.tight_layout()
plt.savefig("outputs/ebrains_eda/09_tissue_area_by_idh.png")
plt.close()

# -- 10. Summary table ---------------------------------------------------------
summary = pd.DataFrame({
    "Total cases": [len(df_deduped)],
    "IDH mutant": [(df_deduped["idh_status"] == "mutant").sum()],
    "IDH wildtype": [(df_deduped["idh_status"] == "wildtype").sum()],
    "GBM": [(df_deduped["tumour_type"] == "GBM").sum()],
    "Astrocytoma": [(df_deduped["tumour_type"] == "astrocytoma").sum()],
    "Oligodendroglioma": [(df_deduped["tumour_type"] == "oligodendroglioma").sum()],
    "Male": [(df_deduped["sex"] == "male").sum()],
    "Female": [(df_deduped["sex"] == "female").sum()],
    "Mean age": [round(df_deduped["age"].mean(), 1)],
    "Cerebral": [df_deduped["is_cerebral"].sum()],
    "Missing age": [df_deduped["age"].isna().sum()],
    "Missing sex": [df_deduped["sex"].isna().sum()],
}).T.rename(columns={0: "value"})

summary.to_csv("outputs/ebrains_eda/summary_table.csv")
print("\nSummary table:")
print(summary.to_string())

print("\n" + "=" * 60)
print(f"All plots saved to outputs/ebrains_eda/")
print("=" * 60)