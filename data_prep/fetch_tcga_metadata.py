# scripts/fetch_tcga_all.py
"""
Fetch all TCGA-GBM and TCGA-LGG metadata.
IDH labels: Mendonça et al. 2025 (WHO 2021) primary, Ceccarelli et al. 2016 fallback.
Dedup rule: one DX slide per case  largest file wins.
Size QC:    flag slides < 50 MB as suspicious.

Outputs to data/raw/tcga/clinical/
  - tcga_demographics.tsv
  - tcga_slides.tsv
  - tcga_slides_clean.csv      <- all slides, not deduped
  - tcga_idh_final.csv
  - tcga_clinical.csv          <- one row per case, deduped slide info
  - tcga_manifest.csv          <- download manifest (usable cases only)
  manifests/tcga_gdc_manifest.txt  <- file IDs for gdc-client
"""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path

Path("data/raw/tcga/clinical").mkdir(parents=True, exist_ok=True)
Path("manifests").mkdir(parents=True, exist_ok=True)

GDC = "https://api.gdc.cancer.gov"

#  1. DEMOGRAPHICS 
print("=" * 60)
print("1. Fetching demographics from GDC...")

filters = {
    "op": "in",
    "content": {
        "field": "project.project_id",
        "value": ["TCGA-GBM", "TCGA-LGG"]
    }
}

fields = [
    "submitter_id",
    "project.project_id",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.gender",
    "demographic.age_at_index",
    "demographic.vital_status",
    "diagnoses.primary_diagnosis",
    "diagnoses.tumor_grade",
    "diagnoses.morphology",
    "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.site_of_resection_or_biopsy",
    "tissue_source_site.code",
    "tissue_source_site.name",
]

r = requests.get(f"{GDC}/cases", params={
    "filters": json.dumps(filters),
    "fields":  ",".join(fields),
    "size":    2000,
    "format":  "TSV",
})
print(f"  Status: {r.status_code}")

with open("data/raw/tcga/clinical/tcga_demographics.tsv", "wb") as f:
    f.write(r.content)

df_demo = pd.read_csv("data/raw/tcga/clinical/tcga_demographics.tsv", sep="\t")
df_demo["case_id"] = df_demo["submitter_id"].str[:12]

print(f"  Cases: {len(df_demo)}")
print(f"\n  Project breakdown:")
print(df_demo["project.project_id"].value_counts().to_string())
print(f"\n  Race:")
print(df_demo["demographic.race"].value_counts(dropna=False).to_string())
print(f"\n  Gender:")
print(df_demo["demographic.gender"].value_counts(dropna=False).to_string())
print(f"\n  Age stats:")
print(df_demo["demographic.age_at_index"].describe().to_string())
print(f"\n  Tissue source sites: {df_demo['tissue_source_site.code'].nunique()}")
print(f"\n  Top 10 sites:")
print(df_demo["tissue_source_site.name"].value_counts().head(10).to_string())

#  2. DIAGNOSTIC SLIDES 
print("\n" + "=" * 60)
print("2. Fetching diagnostic slide metadata from GDC...")

slide_filters = {
    "op": "and",
    "content": [
        {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": ["TCGA-GBM", "TCGA-LGG"]
            }
        },
        {"op": "=", "content": {"field": "data_type",             "value": "Slide Image"}},
        {"op": "=", "content": {"field": "data_format",           "value": "SVS"}},
        {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
    ]
}

slide_fields = [
    "file_id", "file_name", "file_size",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.tissue_source_site.code",
    "cases.tissue_source_site.name",
    "cases.samples.sample_type",
    "cases.samples.portions.slides.section_location",
    "cases.samples.portions.slides.percent_tumor_cells",
]

r2 = requests.get(f"{GDC}/files", params={
    "filters": json.dumps(slide_filters),
    "fields":  ",".join(slide_fields),
    "size":    3000,
    "format":  "TSV",
})
print(f"  Status: {r2.status_code}")

with open("data/raw/tcga/clinical/tcga_slides.tsv", "wb") as f:
    f.write(r2.content)

df_slides = pd.read_csv("data/raw/tcga/clinical/tcga_slides.tsv", sep="\t")

def first_col(df, keyword):
    cols = [c for c in df.columns if keyword in c]
    return df[cols[0]] if cols else pd.Series([None] * len(df))

df_slides_clean = pd.DataFrame({
    "file_id":   df_slides["file_id"],
    "file_name": df_slides["file_name"],
    "file_size": df_slides["file_size"],
    "case_id":   first_col(df_slides, "submitter_id").str[:12],
    "project":   first_col(df_slides, "project_id"),
    "site_code": first_col(df_slides, "tissue_source_site.code"),
    "site_name": first_col(df_slides, "tissue_source_site.name"),
})

df_slides_clean["is_dx"]  = df_slides_clean["file_name"].str.contains(
    r"-DX\d", case=False, regex=True
)
df_slides_clean["size_mb"] = df_slides_clean["file_size"] / 1e6

df_slides_clean.to_csv("data/raw/tcga/clinical/tcga_slides_clean.csv", index=False)

df_dx = df_slides_clean[df_slides_clean["is_dx"]].copy()

print(f"\n  All slides:           {len(df_slides_clean)}")
print(f"  DX slides:            {len(df_dx)}")
print(f"  Non-DX:               {(~df_slides_clean['is_dx']).sum()}")
print(f"  Unique cases with DX: {df_dx['case_id'].nunique()}")

#  dedup: one DX slide per case (largest) 
multi = df_dx.groupby("case_id").size()
print(f"\n  Cases with >1 DX slide: {(multi > 1).sum()}")
print(f"  Distribution of DX slides per case:")
print(multi.value_counts().sort_index().to_string())

df_dx_dedup = (
    df_dx
    .sort_values("file_size", ascending=False)
    .drop_duplicates(subset="case_id", keep="first")
    .copy()
)
print(f"\n  After dedup (one per case): {len(df_dx_dedup)}")

#  size QC 
SIZE_THRESHOLD_MB = 50
df_dx_dedup["size_flag"] = df_dx_dedup["size_mb"] < SIZE_THRESHOLD_MB
n_small = df_dx_dedup["size_flag"].sum()

print(f"\n  Size QC (< {SIZE_THRESHOLD_MB} MB flagged as suspicious):")
print(f"  Flagged slides: {n_small}")
if n_small > 0:
    print(df_dx_dedup[df_dx_dedup["size_flag"]][
        ["file_name", "case_id", "project", "size_mb"]
    ].sort_values("size_mb").to_string(index=False))

print(f"\n  Deduped DX size stats (MB):")
print(df_dx_dedup["size_mb"].describe().to_string())
print(f"  Total download size (all deduped DX): "
      f"{df_dx_dedup['size_mb'].sum()/1e3:.1f} GB")

#  3. IDH STATUS 
print("\n" + "=" * 60)
print("3. Building IDH labels...")

# 3a. Mendonça 2025 WHO 2021 (primary)
print("\n  3a. Mendonça et al. 2025 (WHO 2021)...")
df_who = pd.read_csv("data/raw/tcga/clinical/Matrix_WHO2021.csv")
df_who["case_id"] = df_who["Patient_ID"].str[:12]

def parse_idh_who(label):
    label = str(label).lower()
    if "idhwt"  in label: return "wildtype"
    if "idhmut" in label: return "mutant"
    return None

df_who["idh_status_who2021"]  = df_who["classification.2021_complete.labels"].apply(parse_idh_who)
df_who["tumour_type_who2021"] = df_who["classification.2021_simplified.labels"]
df_who["label_who2021_full"]  = df_who["classification.2021_complete.labels"]

print(f"  Cases: {len(df_who)}")
print(f"  IDH status (WHO 2021):")
print(df_who["idh_status_who2021"].value_counts(dropna=False).to_string())
print(f"  Tumour type (WHO 2021):")
print(df_who["tumour_type_who2021"].value_counts(dropna=False).to_string())

who_lookup = df_who.set_index("case_id")[
    ["idh_status_who2021", "tumour_type_who2021", "label_who2021_full"]
].to_dict("index")

# 3b. Ceccarelli 2016 (fallback)
print("\n  3b. Ceccarelli et al. 2016 (fallback)...")
df_c = pd.read_excel(
    "data/raw/tcga/clinical/mmc2.xlsx",
    sheet_name="S1A. TCGA discovery dataset",
    engine="openpyxl",
    skiprows=1
)
df_c["case_id"] = df_c["Case"].str[:12]
df_c["idh_status_ceccarelli"] = df_c["IDH status"].str.strip().map(
    {"Mutant": "mutant", "WT": "wildtype"}
)
df_c["idh_codel"] = df_c["IDH/codel subtype"].str.strip()

cec_lookup = df_c.set_index("case_id")[
    ["idh_status_ceccarelli", "idh_codel",
     "MGMT promoter status", "ATRX status",
     "TERT promoter status", "Chr 7 gain/Chr 10 loss",
     "Grade", "Histology"]
].to_dict("index")

print(f"  Cases: {len(df_c)}")
print(f"  Ceccarelli IDH:")
print(df_c["idh_status_ceccarelli"].value_counts(dropna=False).to_string())

# 3c. Combine
print("\n  3c. Combining...")
idh_records = []
for _, row in df_demo.iterrows():
    cid     = row["case_id"]
    project = row["project.project_id"]
    who     = who_lookup.get(cid, {})
    cec     = cec_lookup.get(cid, {})
    who_status = who.get("idh_status_who2021")
    cec_status = cec.get("idh_status_ceccarelli")

    if who_status is not None:
        final_status = who_status
        source       = "Mendonca2025_WHO2021"
    elif cec_status is not None:
        final_status = cec_status
        source       = "Ceccarelli2016"
    else:
        final_status = None
        source       = "missing"

    idh_records.append({
        "case_id":                cid,
        "project_id":             project,
        "idh_status":             final_status,
        "idh_source":             source,
        "tumour_type_who2021":    who.get("tumour_type_who2021"),
        "label_who2021_full":     who.get("label_who2021_full"),
        "idh_codel":              cec.get("idh_codel"),
        "MGMT promoter status":   cec.get("MGMT promoter status"),
        "ATRX status":            cec.get("ATRX status"),
        "TERT promoter status":   cec.get("TERT promoter status"),
        "Chr 7 gain/Chr 10 loss": cec.get("Chr 7 gain/Chr 10 loss"),
        "Grade":                  cec.get("Grade"),
        "Histology":              cec.get("Histology"),
    })

df_idh = pd.DataFrame(idh_records)
df_idh["idh_status"] = df_idh["idh_status"].where(
    df_idh["idh_status"].notna(), other=pd.NA
)

print(f"\n  IDH status distribution:")
print(df_idh["idh_status"].value_counts(dropna=False).to_string())
print(f"\n  IDH by project:")
print(df_idh.groupby(["project_id", "idh_status"], dropna=False).size().to_string())
print(f"\n  Source breakdown:")
print(df_idh["idh_source"].value_counts().to_string())
print(f"\n  Missing IDH: {df_idh['idh_status'].isna().sum()}")

df_idh.to_csv("data/raw/tcga/clinical/tcga_idh_final.csv", index=False)
print(f"\n  Saved tcga_idh_final.csv")

#  4. MASTER TABLE (one row per case, deduped slide info) 
print("\n" + "=" * 60)
print("4. Building master clinical table (deduped)...")

# slide info from deduped DX table  one row per case
dx_info = df_dx_dedup[[
    "case_id", "file_id", "file_name", "file_size",
    "size_mb", "size_flag", "site_code", "site_name"
]].rename(columns={
    "file_id":   "dx_file_id",
    "file_name": "dx_file_name",
    "file_size": "dx_file_size",
    "size_mb":   "dx_size_mb",
    "size_flag": "dx_size_flag",
    "site_code": "dx_site_code",
    "site_name": "dx_site_name",
})

# also carry total slide count per case for reference
n_dx_per_case = df_dx.groupby("case_id").size().reset_index(name="n_dx_slides_total")

df_merged = (
    df_demo
    .merge(df_idh[[
        "case_id", "idh_status", "idh_source",
        "tumour_type_who2021", "label_who2021_full", "idh_codel",
        "MGMT promoter status", "ATRX status",
        "TERT promoter status", "Chr 7 gain/Chr 10 loss",
        "Grade", "Histology",
    ]], on="case_id", how="left")
    .merge(dx_info,          on="case_id", how="left")
    .merge(n_dx_per_case,    on="case_id", how="left")
)

# usable: has IDH label + has DX slide + slide not flagged as too small
df_merged["has_dx"]   = df_merged["dx_file_id"].notna()
df_merged["usable"]   = (
    df_merged["idh_status"].notna() &
    df_merged["has_dx"] &
    ~df_merged["dx_size_flag"].fillna(False)
)

print(f"  Total cases:                {len(df_merged)}")
print(f"  Has DX slide (deduped):     {df_merged['has_dx'].sum()}")
print(f"  Missing IDH:                {df_merged['idh_status'].isna().sum()}")
print(f"  DX slide too small (<50MB): {df_merged['dx_size_flag'].fillna(False).sum()}")
print(f"  Usable cases:               {df_merged['usable'].sum()}")

df_usable = df_merged[df_merged["usable"]]
print(f"\n  Usable IDH by project:")
print(df_usable.groupby(
    ["project.project_id", "idh_status"]
).size().to_string())

print(f"\n  Race distribution in usable set:")
print(df_usable["demographic.race"].value_counts(dropna=False).to_string())

print(f"\n  IDH status:")
print(df_usable["idh_status"].value_counts().to_string())

print(f"\n  Download size (usable only): "
      f"{df_usable['dx_size_mb'].sum()/1e3:.1f} GB")

df_merged.to_csv("data/raw/tcga/clinical/tcga_clinical.csv", index=False)
print(f"\n  Saved tcga_clinical.csv ({len(df_merged)} rows, one per case)")

#  5. DOWNLOAD MANIFEST 
print("\n" + "=" * 60)
print("5. Building download manifest...")

df_manifest = df_usable[[
    "case_id", "project.project_id",
    "dx_file_id", "dx_file_name", "dx_size_mb",
    "idh_status", "idh_source", "tumour_type_who2021",
    "demographic.race", "demographic.gender",
    "demographic.age_at_index", "tissue_source_site.name",
    "n_dx_slides_total",
]].rename(columns={
    "project.project_id":        "project",
    "dx_file_id":                "file_id",
    "dx_file_name":              "file_name",
    "dx_size_mb":                "size_mb",
    "demographic.race":          "race",
    "demographic.gender":        "sex",
    "demographic.age_at_index":  "age",
    "tissue_source_site.name":   "site",
})

df_manifest.to_csv("data/raw/tcga/clinical/tcga_manifest.csv", index=False)
print(f"  Saved tcga_manifest.csv ({len(df_manifest)} usable cases)")

# GDC-client format: just file IDs, one per line
df_manifest[["file_id"]].to_csv(
    "manifests/tcga_gdc_manifest.txt",
    index=False, header=True
)
print(f"  Saved manifests/tcga_gdc_manifest.txt")

print(f"\n  Manifest breakdown:")
print(df_manifest.groupby(["project", "idh_status"]).size().to_string())

print("\n" + "=" * 60)
print("DONE.")
print("  data/raw/tcga/clinical/tcga_demographics.tsv")
print("  data/raw/tcga/clinical/tcga_slides.tsv")
print("  data/raw/tcga/clinical/tcga_slides_clean.csv  <- all slides")
print("  data/raw/tcga/clinical/tcga_idh_final.csv")
print("  data/raw/tcga/clinical/tcga_clinical.csv      <- one row per case, deduped")
print("  data/raw/tcga/clinical/tcga_manifest.csv      <- usable cases for download")
print("  manifests/tcga_gdc_manifest.txt               <- file IDs for gdc-client")