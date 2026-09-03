# scripts/label_confidence_tiers_ipd_brain.py
"""
Ordinal IDH-label confidence tiers for IPD Brain, built entirely from columns
already in data/raw/ipd_brain/clinical/ipd_brain_v1.csv -- no new data needed.

IHC (IDH1R132H) only catches the single most common IDH mutation and misses
non-R132H IDH1/IDH2 mutants (~5-10% of IDH-mutant gliomas). Mutant calls are
trusted as-is (IHC is highly specific for R132H). WT calls are split into two
confidence levels:

  Level A -- high confidence: mutant, or WT with no case-specific evidence of
             mislabeling.
  Level C -- case-specific evidence of likely mislabeling: WT, and at least
             one of:
               c1  diagnosis text mentions "oligodendroglioma" -- WHO requires
                   IDH-mutant + 1p/19q-codeletion for that diagnosis, so a
                   WT-labeled oligodendroglioma is internally contradictory.
               c2  ATRX loss AND p53 diffuse-positive together -- the classic
                   IDH-mutant-astrocytoma phenotype despite negative R132H
                   (Reuss et al. 2015, Acta Neuropathol, PMID 25427834: 137/141
                   ATRX-loss diffuse gliomas were IDH-mutant; Melguizo-Gavilanes
                   et al. 2015, PMID 26395639: ATRX loss correlates with both
                   IDH1R132H mutation and p53 overexpression).
               c4  diagnosis text itself says "mutant" despite the WT IHC call
                   -- a direct textual contradiction (e.g. IN Brain-0153... see
                   IN Brain-0012 specifically).


Usage:
    python scripts/label_confidence_tiers_ipd_brain.py
"""

import re
from pathlib import Path

import pandas as pd

CLINICAL_CSV = Path("data/raw/ipd_brain/clinical/ipd_brain_v1.csv")
CURATION_CSV = Path("outputs/ipd_brain_curation/patient_level_curation.csv")
OUT_CSV = Path("outputs/ipd_brain_curation/label_confidence_tiers.csv")

YOUNG_AGE_CUTOFF = 55   # WHO2016/Chen et al. 2014: reflex sequencing recommended below this age
GBM_GRADE = 4

SLIDE_SUFFIX_RE = re.compile(r"\([a-z]\)$", re.IGNORECASE)


def to_patient_id(case_number: str) -> str:
    """Same derivation as check_ipd_brain_exclusions.py: first listed slide id,
    with any (a)/(b) suffix stripped."""
    first = str(case_number).split("\n")[0].strip()
    return SLIDE_SUFFIX_RE.sub("", first).strip()


def main():
    clinical = pd.read_csv(CLINICAL_CSV, encoding="utf-8-sig")
    clinical["patient_id"] = clinical["Case Number"].apply(to_patient_id)
    clinical = clinical.drop_duplicates("patient_id")

    curation = pd.read_csv(CURATION_CSV)
    included_ids = set(curation.loc[curation["status"] == "included", "patient_id"])
    df = clinical[clinical["patient_id"].isin(included_ids)].copy()

    missing = included_ids - set(df["patient_id"])
    if missing:
        raise RuntimeError(
            f"{len(missing)} included patient(s) not found in clinical CSV: {sorted(missing)}"
        )

    is_mutant = df["IDH1R132H"] == 1
    is_wt = df["IDH1R132H"] == 0
    if not (is_mutant | is_wt).all():
        bad = df.loc[~(is_mutant | is_wt), "patient_id"].tolist()
        raise RuntimeError(f"Non-binary/missing IDH1R132H for included patient(s): {bad}")

    diagnosis = df["Diagnosis"].astype(str)
    c1_oligo = diagnosis.str.contains("oligodendroglioma", case=False, na=False)
    c2_atrx_p53 = (df["ATRX"] == "Not Retained") & (df["p53"] == 1)
    c3_young_gbm = (df["Age"] < YOUNG_AGE_CUTOFF) & (df["WHO Grade"] == GBM_GRADE)
    c4_text_mutant = diagnosis.str.contains("mutant", case=False, na=False)

    evidence = is_wt & (c1_oligo | c2_atrx_p53 | c4_text_mutant)
    high_confidence = is_mutant | (is_wt & ~evidence)

    assert (high_confidence.astype(int) + evidence.astype(int) == 1).all(), \
        "levels must partition every patient exactly once"

    df["level"] = "A"
    df.loc[evidence, "level"] = "C"

    df["c1_oligo_diagnosis"] = c1_oligo
    df["c2_atrx_loss_and_p53_positive"] = c2_atrx_p53
    df["c3_young_and_gbm"] = c3_young_gbm
    df["c4_diagnosis_says_mutant"] = c4_text_mutant

    out_cols = [
        "patient_id", "Age", "WHO Grade", "Diagnosis", "IDH1R132H", "ATRX", "p53",
        "level", "c1_oligo_diagnosis", "c2_atrx_loss_and_p53_positive",
        "c3_young_and_gbm", "c4_diagnosis_says_mutant",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(OUT_CSV, index=False)

    n = len(df)
    print(f"Total included patients: {n}")
    print(f"  Mutant (IHC+):              {int(is_mutant.sum())}")
    print(f"  WT (IHC-):                  {int(is_wt.sum())}")
    print()
    for level, label in [("A", "High confidence"),
                         ("C", "Case-specific evidence of likely mislabeling")]:
        cnt = int((df["level"] == level).sum())
        print(f"Level {level} ({label}): {cnt} ({100 * cnt / n:.0f}%)")

    print("\nLevel C breakdown (a patient can trigger more than one):")
    c_df = df[df["level"] == "C"]
    print(f"  c1 oligodendroglioma diagnosis: {int(c_df['c1_oligo_diagnosis'].sum())}")
    print(f"  c2 ATRX loss AND p53+:          {int(c_df['c2_atrx_loss_and_p53_positive'].sum())}")
    print(f"  c4 diagnosis says 'mutant':     {int(c_df['c4_diagnosis_says_mutant'].sum())}")

    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
