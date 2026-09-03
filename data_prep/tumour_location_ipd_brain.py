# scripts/tumour_location_ipd_brain.py
"""
Normalised anatomical tumour-location bins for IPD Brain, built entirely from
the free-text `SITE` column in data/raw/ipd_brain/clinical/ipd_brain_v1.csv --
no new data needed.

`SITE` is unusable as a subgroup variable as-is: 188 distinct free-text
strings across 328 patients ("Frontal", "Left frontal", "Left frontal lobe",
"? site", multi-lobe combinations, inconsistent capitalisation/typos), i.e.
close to one value per patient. This collapses it into a handful of broad
anatomical bins so it can actually be tested for OOD-score/subgroup
heterogeneity (see ood_evaluation_ipd_brain_extension.ipynb) without every
group falling below MIN_GROUP_N.

Label-independence note: this variable has nothing to do with IDH status --
it does not need `label_confidence_tiers_ipd_brain.py`'s tier restriction.
Runs on the full included cohort.

Normalisation logic, applied per patient:
  1. Lowercase, strip laterality (left/right/bilateral/...) and noise words
     (lobe/region/lesion/cortex/gyrus/...), fix known typos (e.g.
     "tempral"->"temporal", "collosum"->"callosum").
  2. A handful of unambiguous phrase-level overrides checked first (motor
     cortex/precentral -> Frontal; cingulate gyrus, thalamus, corpus
     callosum, cerebellum, ventricle, falx, suprasellar -> Deep / midline;
     "?"/blank/"brain"/stereotactic biopsy/uninterpretable strings ->
     Unknown / not recorded).
  3. Remaining text is matched against 5 primary cortical-lobe keywords
     (frontal/temporal/parietal/occipital/insular). Exactly one match ->
     that lobe. Two or more (e.g. "fronto-parietal", "parieto-occipital")
     -> Multi-lobe / junctional, not force-assigned to either lobe alone.
     No match at all -> Unknown / not recorded.

Every patient gets a bin (nothing silently dropped); MIN_GROUP_N=15 (same
threshold ood_evaluation_analysis.ipynb already uses, for the same reason:
below this a two-sample AUROC/heterogeneity test isn't informative) is
applied only at the *reporting* stage, listing under-powered bins rather
than excluding them from the output CSV.

Usage:
    python scripts/tumour_location_ipd_brain.py
"""

import re
from pathlib import Path

import pandas as pd

CLINICAL_CSV = Path("data/raw/ipd_brain/clinical/ipd_brain_v1.csv")
CURATION_CSV = Path("outputs/ipd_brain_curation/patient_level_curation.csv")
OUT_CSV = Path("outputs/ipd_brain_curation/tumour_location_bins.csv")

MIN_GROUP_N = 15   # matches ood_evaluation_analysis.ipynb's pre-registered threshold

SLIDE_SUFFIX_RE = re.compile(r"\([a-z]\)$", re.IGNORECASE)

TYPO_FIXES = [
    (r"\bprietal\b", "parietal"),
    (r"\bperietal\b", "parietal"),
    (r"\btempral\b", "temporal"),
    (r"\bteporal\b", "temporal"),
    (r"\binvasular\b", "insular"),
    (r"\bcollosum\b", "callosum"),
    (r"\bluperior\b", "superior"),
]

NOISE_WORDS = [
    "brain", "lobe", "lobule", "region", "lesion", "sol", "cortex", "gyrus",
    "area", "site", "stereotactic", "biopsy", "and", "of",
]

LATERALITY = ["left", "right", "lt", "rt", "bilateral", "bihemispheric"]

# Checked against the RAW lowercased string (before noise-word stripping,
# since some phrases -- "motor cortex", "cingulate gyrus" -- use words that
# are stripped as noise for the generic lobe-keyword pass below).
RAW_OVERRIDES = [
    (re.compile(r"motor cortex|premotor|precentral|post ?central motor"), "Frontal"),
    (re.compile(r"cg region|cingulate"), "Deep / midline"),
    (re.compile(r"supracellar|suprasellar"), "Deep / midline"),
    (re.compile(r"falx"), "Deep / midline"),
    (re.compile(r"lateral ventricle"), "Deep / midline"),
    (re.compile(r"stereotactic biopsy"), "Unknown / not recorded"),
    (re.compile(r"^\s*\??\s*site\s*$|^\s*brain,?\s*$|^\s*$|^\s*nan\s*$"), "Unknown / not recorded"),
    (re.compile(r"multicentric|frontotemporoparietal"), "Multi-lobe / junctional"),
]

LOBE_PATTERNS = {
    # Stems, not full words: "parieto-occipital"/"fronto-temporal"-style combining
    # forms don't contain the full word ("parieto" has no "parietal" substring), so
    # matching only full words silently collapsed multi-lobe strings like
    # "right parieto occipital" into a single lobe. Stems catch both the full word
    # and the combining form.
    "Frontal": re.compile(r"front"),
    "Temporal": re.compile(r"tempor|mesial temporal|medial temporal|sylvian|hippocampus"),
    "Parietal": re.compile(r"pariet|paracentral|postcentral"),
    "Occipital": re.compile(r"occipit"),
    "Insular": re.compile(r"insul"),
}
DEEP_PATTERNS = {
    "thalamus": re.compile(r"thalam"),
    "corpus_callosum": re.compile(r"corpus call"),
    "cerebellum": re.compile(r"cerebell"),
}


def to_patient_id(case_number: str) -> str:
    """Same derivation as check_ipd_brain_exclusions.py / label_confidence_tiers_ipd_brain.py:
    first listed slide id, with any (a)/(b) suffix stripped."""
    first = str(case_number).split("\n")[0].strip()
    return SLIDE_SUFFIX_RE.sub("", first).strip()


def clean(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in ("nan", ""):
        return ""
    s = s.replace(",", " ").replace(".", " ").replace("&", " ")
    for pat, rep in TYPO_FIXES:
        s = re.sub(pat, rep, s)
    for w in LATERALITY:
        s = re.sub(rf"\b{w}\b", " ", s)
    for w in NOISE_WORDS:
        s = re.sub(rf"\b{w}\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def classify(raw: str) -> tuple[str, str]:
    raw_lower = str(raw).strip().lower()
    for pat, label in RAW_OVERRIDES:
        if pat.search(raw_lower):
            return clean(raw), label

    cleaned = clean(raw)
    if cleaned == "":
        return cleaned, "Unknown / not recorded"

    matched_lobes = [name for name, pat in LOBE_PATTERNS.items() if pat.search(cleaned)]
    matched_deep = [name for name, pat in DEEP_PATTERNS.items() if pat.search(cleaned)]

    if len(matched_lobes) == 1 and not matched_deep:
        return cleaned, matched_lobes[0]
    if len(matched_lobes) >= 2 or (matched_lobes and matched_deep):
        return cleaned, "Multi-lobe / junctional"
    if matched_deep:
        return cleaned, "Deep / midline"
    return cleaned, "Unknown / not recorded"


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

    results = df["SITE"].apply(classify)
    df["site_cleaned"] = [r[0] for r in results]
    df["location_bin"] = [r[1] for r in results]

    out_cols = ["patient_id", "SITE", "site_cleaned", "location_bin"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].rename(columns={"SITE": "site_raw"}).to_csv(OUT_CSV, index=False)

    n = len(df)
    counts = df["location_bin"].value_counts()
    print(f"Total included patients: {n}\n")
    print("Location bin counts (Unknown / not recorded is not a real subgroup -- exclude it")
    print(f"from any heterogeneity test, don't treat it as a location):")
    for label, cnt in counts.items():
        flag = "" if cnt >= MIN_GROUP_N or label == "Unknown / not recorded" else \
            f"  <-- below MIN_GROUP_N={MIN_GROUP_N}, list in dropped_groups, don't test alone"
        print(f"  {label:28s} {cnt:4d} ({100*cnt/n:4.1f}%){flag}")

    testable = counts[(counts.index != "Unknown / not recorded") & (counts >= MIN_GROUP_N)]
    print(f"\n{len(testable)} bins meet MIN_GROUP_N and are testable: {list(testable.index)}")
    under_powered = counts[(counts.index != "Unknown / not recorded") & (counts < MIN_GROUP_N)]
    if len(under_powered):
        print(f"Under-powered (kept in the CSV, not deleted, just don't test alone): "
              f"{list(under_powered.index)}")

    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
