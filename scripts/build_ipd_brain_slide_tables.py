"""
Builds data/processed/slide_table_ipd_brain_{encoder}.csv (PATIENT, FILENAME),
matching the TCGA/EBRAINS convention exactly, plus a clinical CSV
(data/raw/ipd_brain/clinical/ipd_brain_idh_final.csv) with idh_status and the
label confidence tier -- so the existing extract_abmil_embeddings.py /
mc_dropout_infer.py / laplace_infer.py / deep_ensemble_infer.py scripts can
be extended with a simple "ipd_brain" branch in load_slide_table() rather
than needing bespoke IPD Brain scripts.

Uses patient_level_curation.csv (which slide to use, which patients passed
exclusion) + label_confidence_tiers.csv (the verified 3-tier IHC label
confidence system, not the IHC call taken at face value) as the sole
sources of truth 


Usage:
    python scripts/build_ipd_brain_slide_tables.py
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

ENCODER_FEATURE_DIRS = {
    "uni2":     "data/processed/features/ipd_brain/uni/uni2-49b04e14",
    "conch":    "data/processed/features/ipd_brain/conch/conch-49b04e14",
    "hoptimus": "data/processed/features/ipd_brain/hoptimus/h-optimus-1-49b04e14",
}


def build_clinical():
    cur = pd.read_csv(BASE / "outputs/ipd_brain_curation/patient_level_curation.csv")
    tiers = pd.read_csv(BASE / "outputs/ipd_brain_curation/label_confidence_tiers.csv")

    cur = cur[cur["status"] == "included"]
    merged = cur.merge(tiers[["patient_id", "level"]], on="patient_id", how="left")
    assert merged["level"].notna().all(), "every included patient must have a confidence tier"

    out = merged.rename(columns={
        "patient_id": "case_id",
        "kept_slide_id": "FILENAME_STEM",
        "idh1r132h": "idh_status",
    })[["case_id", "FILENAME_STEM", "idh_status", "level"]]
    out["idh_status"] = out["idh_status"].astype(int)


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

    out = out.merge(
        exploded[["case_id_base", "Age", "Sex"]],
        left_on="case_id", right_on="case_id_base", how="left"
    ).drop(columns=["case_id_base"]).rename(columns={"Age": "age", "Sex": "sex"})
    n_missing_demo = out["age"].isna().sum()
    if n_missing_demo:
        print(f"WARNING: {n_missing_demo} patients missing age/sex after ID normalisation")

    out_path = BASE / "data/raw/ipd_brain/clinical/ipd_brain_idh_final.csv"
    out.to_csv(out_path, index=False)
    print(f"clinical: {len(out)} patients "
          f"(level A={sum(out.level=='A')}, B={sum(out.level=='B')}, C={sum(out.level=='C')}), "
          f"idh_status mutant={out.idh_status.sum()} wildtype={(out.idh_status==0).sum()}")
    print(f"  -> wrote {out_path}")
    return out


def build_slide_tables(clinical: pd.DataFrame):
    for encoder, feat_dir in ENCODER_FEATURE_DIRS.items():
        feat_dir = BASE / feat_dir
        rows = []
        missing = []
        for _, row in clinical.iterrows():
            fn = f"{row['FILENAME_STEM']}.h5"
            if not (feat_dir / fn).exists():
                missing.append(fn)
                continue
            rows.append({"PATIENT": row["case_id"], "FILENAME": fn})

        out = pd.DataFrame(rows)
        out_path = BASE / f"data/processed/slide_table_ipd_brain_{encoder}.csv"
        out.to_csv(out_path, index=False)
        print(f"{encoder}: {len(out)} slides, {len(missing)} missing features "
              f"-> wrote {out_path}")
        if missing:
            print(f"  missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")


if __name__ == "__main__":
    clinical = build_clinical()
    build_slide_tables(clinical)
