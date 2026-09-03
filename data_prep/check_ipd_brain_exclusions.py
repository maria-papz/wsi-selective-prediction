# scripts/check_ipd_brain_exclusions.py
"""
IPD-Brain (Chauhan et al., 2024) cohort curation.

Applies the same style of exclusion rules used for EBRAINS (Section 3.2.1)
and TCGA, adapted to what IPD-Brain's clinical CSV actually
records:
  - confirmed IDH status required (IDH1R132H must be 0 or 1)          [EBRAINS-style]
  - post-recurrence-only presentations excluded (no primary sample
    available for that patient in this cohort), detected from
    free-text mentions of "recur" in Diagnosis                        [EBRAINS-style]
  - one slide per patient: where a patient has multiple physical
    slides, the largest file is kept                                  [EBRAINS/TCGA-style]
  - every listed slide must resolve to an actual image file in the
    downloaded archives

Input:  data/raw/ipd_brain/clinical/ipd_brain_v1.csv
        Labeled_Part_{1,2,4}.zip in the project root, plus the
        browser-downloaded Part 3 zip (read via zip central directory
        only -- nothing is extracted)
Output: outputs/ipd_brain_curation/{slide_level,patient_level}.csv
"""

import re
import zipfile
from pathlib import Path

import pandas as pd

CSV_PATH = Path("data/raw/ipd_brain/clinical/ipd_brain_v1.csv")
ZIP_PATHS = {
    1: Path("Labeled_Part_1.zip"),
    2: Path("Labeled_Part_2.zip"),
    3: Path(
        "170acc68-1288-499e-9a91-b951e569e70d_ac676a96-1f94-4c52-a83a-2eb204f71c91"
        "_DATASET-FILE_Labeled_Part_3_zip_20241226073023473.zip"
    ),
    4: Path("Labeled_Part_4.zip"),
}
OUT = Path("outputs/ipd_brain_curation")
OUT.mkdir(parents=True, exist_ok=True)

RECURRENCE_RE = re.compile(r"recur", re.IGNORECASE)
SLIDE_SUFFIX_RE = re.compile(r"\([a-z]\)$", re.IGNORECASE)

# Slides that exist in the archives and have a resolvable (or fallback) MPP, but fail
# for an unrelated infrastructure reason discovered during STAMP preprocessing.
UNPROCESSABLE_SLIDES = {
    "IN Brain-0337": (
        "not a valid pyramidal WSI (OpenSlide reports vendor=None, falls back to a "
        "generic single-resolution reader); 126976x253952px needs ~97GB RAM to tile at "
        "full resolution, exceeding this machine's 60GB -- DataLoader worker OOM-killed "
        "on every encoder (UNI2/CONCH/H-optimus) after the MPP-fallback fix (2026-08-03) "
        "got it past the earlier MPPExtractionError. Not attempted: re-encoding to a "
        "proper pyramidal format via a streaming tool (e.g. vips) might salvage it, but "
        "wasn't judged worth the effort for 1/321 slides."
    ),
}


WSI_DIR = Path("data/raw/ipd_brain/wsi")


def build_file_index():
    """slide_id (e.g. 'IN Brain-0004(a)') -> (size_bytes, part).

    Originally read the zip archives' central directories (nothing extracted, just
    listing). As of 2026-08 those zips have been deleted after their contents were
    extracted to WSI_DIR -- scan the extracted .tiff files directly instead. Falls
    back to the zips if any are still present (e.g. a fresh checkout that hasn't
    extracted yet), so this keeps working either way.
    """
    index = {}
    if WSI_DIR.exists():
        for p in WSI_DIR.glob("*.tiff"):
            index[p.stem] = (p.stat().st_size, "wsi_dir")

    for part, zpath in ZIP_PATHS.items():
        if not zpath.exists():
            continue
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                slide_id = info.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                index.setdefault(slide_id, (info.file_size, part))

    if not index:
        raise FileNotFoundError(
            f"No slide files found in {WSI_DIR} and no ZIP_PATHS archives present -- "
            "cannot build a file index from either source."
        )
    return index


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    file_index = build_file_index()

    slide_rows = []
    patient_rows = []

    for _, row in df.iterrows():
        slide_ids = [s.strip() for s in str(row["Case Number"]).split("\n") if s.strip()]
        patient_id = SLIDE_SUFFIX_RE.sub("", slide_ids[0]).strip()

        idh_label = row["IDH1R132H"]
        diagnosis = str(row["Diagnosis"])
        is_recurrent = bool(RECURRENCE_RE.search(diagnosis))
        idh_missing = pd.isna(idh_label) or idh_label not in (0, 1)

        sizes = {sid: file_index.get(sid, (None, None))[0] for sid in slide_ids}
        available = {
            sid: sz for sid, sz in sizes.items()
            if sz is not None and sid not in UNPROCESSABLE_SLIDES
        }

        if idh_missing:
            patient_reason = "missing/invalid IDH1R132H label"
        elif is_recurrent:
            patient_reason = "post-recurrence-only presentation (no primary slide in cohort)"
        elif not available:
            if any(sid in UNPROCESSABLE_SLIDES for sid in slide_ids):
                patient_reason = "all listed slides unprocessable (see per-slide reason)"
            else:
                patient_reason = "no image file found for any listed slide"
        else:
            patient_reason = None

        kept_sid = max(available, key=available.get) if available else None

        for sid in slide_ids:
            if sid in UNPROCESSABLE_SLIDES:
                slide_status, slide_reason = "excluded", UNPROCESSABLE_SLIDES[sid]
            elif patient_reason is not None:
                slide_status, slide_reason = "excluded", patient_reason
            elif sid not in available:
                slide_status, slide_reason = "excluded", "image file not found in archives"
            elif sid == kept_sid:
                slide_status, slide_reason = "included", None
            else:
                slide_status = "excluded"
                slide_reason = (
                    f"smaller duplicate slide for {patient_id} "
                    f"(kept {kept_sid}, {available[kept_sid] / 1e6:.0f}MB "
                    f"vs {sizes[sid] / 1e6:.0f}MB)"
                )
            slide_rows.append({
                "patient_id": patient_id,
                "slide_id": sid,
                "file_size_mb": None if sizes[sid] is None else round(sizes[sid] / 1e6, 1),
                "idh1r132h": idh_label,
                "diagnosis": diagnosis,
                "status": slide_status,
                "reason": slide_reason,
            })

        patient_rows.append({
            "patient_id": patient_id,
            "n_slides": len(slide_ids),
            "kept_slide_id": kept_sid,
            "idh1r132h": idh_label,
            "diagnosis": diagnosis,
            "status": "excluded" if patient_reason else "included",
            "reason": patient_reason,
        })

    slide_df = pd.DataFrame(slide_rows)
    patient_df = pd.DataFrame(patient_rows)
    slide_df.to_csv(OUT / "slide_level_curation.csv", index=False)
    patient_df.to_csv(OUT / "patient_level_curation.csv", index=False)

    excluded = patient_df[patient_df["status"] == "excluded"]
    included = patient_df[patient_df["status"] == "included"]

    print(f"Total patients in CSV: {len(patient_df)}")
    print(f"Excluded: {len(excluded)}")
    for reason, n in excluded["reason"].value_counts().items():
        print(f"  - {reason}: {n}")

    dup_excluded = (slide_df["status"] == "excluded") & slide_df["reason"].str.startswith(
        "smaller duplicate", na=False
    )
    multi_deduped = patient_df[(patient_df["n_slides"] > 1) & (patient_df["status"] == "included")]
    print(f"Additionally, {dup_excluded.sum()} smaller-duplicate slides dropped "
          f"from {len(multi_deduped)} multi-slide patients that were otherwise kept "
          f"(one slide retained per patient)")

    n_mut = (included["idh1r132h"] == 1).sum()
    n_wt = (included["idh1r132h"] == 0).sum()
    n_total = len(included)
    print(f"\nFinal cohort: {n_total} patients, "
          f"{n_wt} IDH-wildtype ({100 * n_wt / n_total:.1f}%), "
          f"{n_mut} IDH-mutant ({100 * n_mut / n_total:.1f}%)")
    print(f"\nWrote {OUT / 'patient_level_curation.csv'}")
    print(f"Wrote {OUT / 'slide_level_curation.csv'}")


if __name__ == "__main__":
    main()
