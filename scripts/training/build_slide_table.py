"""
Build slide_table.csv linking patient/case IDs to feature .h5 files.
STAMP's crossval/train commands expect a CSV with columns:
  PATIENT, FILENAME (relative to feature_dir)

Run from project root:
    source venv/bin/activate
    python scripts/build_slide_table.py --encoder uni2 --dataset ebrains
"""

import argparse
import pandas as pd
from pathlib import Path

BASE = Path(".")

ENCODER_DIRS = {
    "uni2":      ("uni", "uni2-49b04e14"),
    "conch":     ("conch", "conch-49b04e14"),
    "hoptimus":  ("hoptimus", "h-optimus-1-49b04e14"),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=["uni2", "conch", "hoptimus"], required=True)
    parser.add_argument("--dataset", choices=["ebrains", "tcga"], required=True)
    args = parser.parse_args()

    encoder_root, encoder_subdir = ENCODER_DIRS[args.encoder]

    if args.dataset == "ebrains":
        feature_dir = BASE / "data/processed/features/ebrains" / encoder_root / encoder_subdir
        clinical = pd.read_csv(BASE / "data/raw/ebrains/clinical/ebrains_idh_cases.csv")
        case_id_col = "uuid"
    else:
        feature_dir = BASE / "data/processed/features/tcga" / encoder_root / encoder_subdir
        clinical = pd.read_csv(BASE / "data/raw/tcga/clinical/tcga_clinical.csv")
        case_id_col = "case_id"

    # find all h5 files (may be nested in subdirectories for TCGA)
    h5_files = list(feature_dir.rglob("*.h5"))
    print(f"Found {len(h5_files)} h5 files in {feature_dir}")

    rows = []
    for h5_path in h5_files:
        rel_path = h5_path.relative_to(feature_dir)
        filename = h5_path.stem  # slide identifier without extension

        if args.dataset == "ebrains":
            # EBRAINS: h5 filename == case_id (e.g. a1960904-...-ndpi)
            case_id = filename
        else:
            # TCGA: h5 filename is like TCGA-XX-XXXX-01Z-00-DX1.<uuid>
            # case_id is the first 12 chars: TCGA-XX-XXXX
            case_id = filename[:12]

        rows.append({
            "PATIENT": case_id,
            "FILENAME": str(rel_path),
        })

    slide_table = pd.DataFrame(rows)
    print(f"Slide table: {len(slide_table)} rows, {slide_table['PATIENT'].nunique()} unique patients")

    # check overlap with clinical
    clinical_ids = set(clinical[case_id_col].astype(str))
    table_ids = set(slide_table["PATIENT"])
    missing = table_ids - clinical_ids
    if missing:
        print(f"WARNING: {len(missing)} slide IDs not found in clinical table")
        print(f"  examples: {list(missing)[:5]}")

    out_path = BASE / f"data/processed/slide_table_{args.dataset}_{args.encoder}.csv"
    slide_table.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()