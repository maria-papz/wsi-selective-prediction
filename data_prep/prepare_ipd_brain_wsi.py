# scripts/prepare_ipd_brain_wsi.py
"""
Unzip the curated IPD-Brain slides into the same layout used for the other
cohorts (data/raw/<cohort>/wsi/), keeping only the slides that survived
check_ipd_brain_exclusions.py (one kept slide per included patient, 321
total, ~498GB).

Disk quota leaves no room to keep both the four source zips (~663GB) and
the extracted slides (~498GB) at once, so each zip is deleted immediately
after its kept slides are extracted and verified -- never both on disk
together for long. Extraction is a raw byte copy (zip entries are STORED,
not deflated), so no decompression is needed.

Input:  outputs/ipd_brain_curation/slide_level_curation.csv
        Labeled_Part_{1,2,4}.zip + the browser-downloaded Part 3 zip
Output: data/raw/ipd_brain/wsi/<slide_id>.tiff  (321 files)
"""

import shutil
import zipfile
from pathlib import Path

import pandas as pd

CURATION_CSV = Path("outputs/ipd_brain_curation/slide_level_curation.csv")
ZIP_PATHS = {
    1: Path("Labeled_Part_1.zip"),
    2: Path("Labeled_Part_2.zip"),
    3: Path(
        "170acc68-1288-499e-9a91-b951e569e70d_ac676a96-1f94-4c52-a83a-2eb204f71c91"
        "_DATASET-FILE_Labeled_Part_3_zip_20241226073023473.zip"
    ),
    4: Path("Labeled_Part_4.zip"),
}
OUT_DIR = Path("data/raw/ipd_brain/wsi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 64 * 1024 * 1024  # 64MB


def main():
    curation = pd.read_csv(CURATION_CSV)
    kept = curation[curation["status"] == "included"].copy()
    print(f"Kept slides to extract: {len(kept)} "
          f"({kept['file_size_mb'].sum() / 1e3:.1f} GB)")

    kept_ids = set(kept["slide_id"])
    expected_size = dict(zip(kept["slide_id"], kept["file_size_mb"]))

    extracted, skipped, failed = [], [], []

    for part, zpath in ZIP_PATHS.items():
        if not zpath.exists():
            print(f"\nPart {part}: {zpath} not found, skipping "
                  f"(assuming already processed and deleted)")
            continue

        print(f"\n=== Part {part}: {zpath} ===")
        with zipfile.ZipFile(zpath) as zf:
            entries_here = [
                info for info in zf.infolist()
                if info.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0] in kept_ids
            ]
            print(f"  {len(entries_here)} kept slides found in this part")

            for info in entries_here:
                slide_id = info.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                dest = OUT_DIR / f"{slide_id}.tiff"

                if dest.exists() and abs(dest.stat().st_size - info.file_size) < 1024:
                    skipped.append(slide_id)
                    continue

                tmp_dest = dest.with_suffix(".tmp")
                with zf.open(info) as src, open(tmp_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=CHUNK_SIZE)

                actual_size = tmp_dest.stat().st_size
                if actual_size != info.file_size:
                    failed.append((slide_id, "size mismatch after copy"))
                    tmp_dest.unlink()
                    continue

                tmp_dest.rename(dest)
                extracted.append(slide_id)
                print(f"  extracted {slide_id}.tiff "
                      f"({info.file_size / 1e6:.0f} MB)")

        # verify every kept slide expected from this part actually landed
        part_ok = all(
            (OUT_DIR / f"{sid}.tiff").exists()
            for sid in {info.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                        for info in entries_here}
        )
        if part_ok:
            print(f"  All Part {part} kept slides verified on disk -- deleting {zpath}")
            zpath.unlink()
        else:
            print(f"  WARNING: not all Part {part} slides verified -- "
                  f"leaving {zpath} in place, NOT deleting")

    print(f"\nExtracted: {len(extracted)}  Skipped (already present): {len(skipped)}  "
          f"Failed: {len(failed)}")
    if failed:
        for sid, reason in failed:
            print(f"  FAILED {sid}: {reason}")

    on_disk = list(OUT_DIR.glob("*.tiff"))
    total_gb = sum(f.stat().st_size for f in on_disk) / 1e9
    print(f"\n{len(on_disk)} files in {OUT_DIR}, {total_gb:.1f} GB total")
    missing = kept_ids - {f.stem for f in on_disk}
    if missing:
        print(f"WARNING: {len(missing)} kept slides still missing: {sorted(missing)[:10]}")
    else:
        print("All 321 kept slides present and accounted for.")


if __name__ == "__main__":
    main()
