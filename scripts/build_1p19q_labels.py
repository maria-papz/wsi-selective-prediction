"""
Build per-patient 1p/19q codeletion labels for TCGA and EBRAINS, restricted
to IDH-mutant cases (codeletion is only diagnostically meaningful once IDH
status is already known -- IDH-wildtype cases are dropped, not labelled
non-codel).

Reuses the existing IDH clinical tables and slide tables -- no re-embedding,
no new downloads. Mirrors the column conventions of ebrains_idh_cases.csv
(idh_binary/dataset/split-style columns) so the output slots into the same
retraining pipeline as the IDH target.

IPD Brain has no 1p/19q assay column, so codel status is proxied from
Subtype (OLIGODENDROGLIOMA -> codel, ASTROCYTOMA -> non-codel; GLIOBLASTOMA
dropped, effectively always IDH-wildtype under WHO CNS5) the same way
EBRAINS's diagnosis-text proxy is already accepted into this pipeline.
Cross-checked against ATRX staining (loss is ~mutually exclusive with 1p/19q
codeletion): 0/66 oligodendroglioma cases show ATRX loss, a clean split
supporting the proxy. 

Outputs:
  data/raw/tcga/clinical/tcga_1p19q_labels.csv
  data/raw/ebrains/clinical/ebrains_1p19q_labels.csv
  data/raw/ipd_brain/clinical/ipd_brain_1p19q_labels.csv
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def build_tcga():
    slides = pd.read_csv(BASE / "data/processed/slide_table_tcga_uni2.csv")
    patients = set(slides["PATIENT"].unique())

    idh = pd.read_csv(BASE / "data/raw/tcga/clinical/tcga_idh_final.csv")
    idh = idh[idh["case_id"].isin(patients)].drop_duplicates("case_id")

    mut = idh[idh["idh_codel"].isin(["IDHmut-codel", "IDHmut-non-codel"])].copy()
    mut["codel_binary"] = (mut["idh_codel"] == "IDHmut-codel").astype(int)
    mut["codel_status"] = mut["codel_binary"].map({1: "codel", 0: "non-codel"})
    mut["dataset"] = "tcga"

    out = mut[[
        "case_id", "idh_codel", "codel_status", "codel_binary",
        "Grade", "Histology", "dataset",
    ]].rename(columns={"case_id": "PATIENT"})

    n_dropped_wt = (idh["idh_codel"] == "IDHwt").sum()
    n_dropped_missing = idh["idh_codel"].isna().sum()
    print(f"TCGA: {len(out)} IDH-mutant cases labelled "
          f"({out['codel_binary'].sum()} codel / {(out['codel_binary']==0).sum()} non-codel); "
          f"dropped {n_dropped_wt} IDHwt, {n_dropped_missing} missing idh_codel")

    out_path = BASE / "data/raw/tcga/clinical/tcga_1p19q_labels.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> wrote {out_path}")
    return out


def build_ebrains():
    ebr = pd.read_csv(BASE / "data/raw/ebrains/clinical/ebrains_idh_cases.csv")
    mut = ebr[ebr["idh_binary"] == 1].copy()

    is_codel = mut["diagnosis"].str.contains("codeleted", case=False, na=False)
    mut["codel_binary"] = is_codel.astype(int)
    mut["codel_status"] = mut["codel_binary"].map({1: "codel", 0: "non-codel"})

    out = mut[[
        "pat_id", "uuid", "diagnosis", "grade", "codel_status", "codel_binary",
        "dataset", "split",
    ]].rename(columns={"pat_id": "PATIENT"})

    n_dropped_wt = (ebr["idh_binary"] == 0).sum()
    print(f"EBRAINS: {len(out)} IDH-mutant cases labelled "
          f"({out['codel_binary'].sum()} codel / {(out['codel_binary']==0).sum()} non-codel); "
          f"dropped {n_dropped_wt} IDHwt")

    out_path = BASE / "data/raw/ebrains/clinical/ebrains_1p19q_labels.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> wrote {out_path}")
    return out


def build_ipd_brain():
    idh = pd.read_csv(BASE / "data/raw/ipd_brain/clinical/ipd_brain_idh_final.csv")
    mut = idh[idh["idh_status"] == 1].copy()
    mut["case_id_base"] = mut["case_id"].str.replace(r"\([a-zA-Z]\)$", "", regex=True).str.strip()

    raw = pd.read_csv(BASE / "data/raw/ipd_brain/clinical/ipd_brain_v1.csv")
    raw.columns = [c.strip().lstrip("﻿") for c in raw.columns]
    raw = raw.rename(columns={"Case Number": "patient_id"})
    raw["patient_id"] = raw["patient_id"].astype(str).str.split("\n")
    exploded = raw.explode("patient_id")
    exploded["patient_id"] = exploded["patient_id"].str.strip()
    exploded["case_id_base"] = (
        exploded["patient_id"].str.replace(r"\([a-zA-Z]\)$", "", regex=True).str.strip()
    )
    exploded = exploded.drop_duplicates("case_id_base")

    mut = mut.merge(
        exploded[["case_id_base", "Subtype", "ATRX"]], on="case_id_base", how="left"
    )
    n_dropped_missing = mut["Subtype"].isna().sum()

    sub = mut[mut["Subtype"].isin(["ASTROCYTOMA", "OLIGODENDROGLIOMA"])].copy()
    sub["codel_binary"] = (sub["Subtype"] == "OLIGODENDROGLIOMA").astype(int)
    sub["codel_status"] = sub["codel_binary"].map({1: "codel", 0: "non-codel"})
    sub["dataset"] = "ipd_brain"

    n_dropped_gbm = (mut["Subtype"] == "GLIOBLASTOMA").sum()

    out = sub[[
        "case_id", "FILENAME_STEM", "Subtype", "ATRX", "codel_status",
        "codel_binary", "level", "dataset",
    ]]

    print(f"IPD Brain: {len(out)} IDH-mutant cases labelled "
          f"({out['codel_binary'].sum()} codel / {(out['codel_binary']==0).sum()} non-codel); "
          f"dropped {n_dropped_gbm} IDH-mutant GBM, {n_dropped_missing} missing Subtype match")

    out_path = BASE / "data/raw/ipd_brain/clinical/ipd_brain_1p19q_labels.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> wrote {out_path}")
    return out


if __name__ == "__main__":
    build_tcga()
    build_ebrains()
    build_ipd_brain()
