# scripts/extract_scanner_metadata.py
"""
Extract scanner metadata from WSI file headers using OpenSlide.
Works on both EBRAINS (NDPI) and TCGA (SVS) slides.
Output: outputs/scanner_metadata/
  - ebrains_scanner_metadata.csv
  - tcga_scanner_metadata.csv
"""

import openslide
import pandas as pd
from pathlib import Path
import traceback

OUT = Path("outputs/scanner_metadata")
OUT.mkdir(parents=True, exist_ok=True)

def extract_scanner_info(wsi_path):
    """Extract scanner metadata from a single WSI."""
    rec = {
        "filename":       wsi_path.name,
        "path":           str(wsi_path),
        "size_mb":        wsi_path.stat().st_size / 1e6,
        "vendor":         None,
        "model":          None,
        "objective_power": None,
        "mpp_x":          None,
        "mpp_y":          None,
        "width":          None,
        "height":         None,
        "n_levels":       None,
        "software":       None,
        "scan_date":      None,
        "status":         "ok",
        "error":          "",
    }
    try:
        slide = openslide.OpenSlide(str(wsi_path))
        props = slide.properties

        rec["width"]    = slide.dimensions[0]
        rec["height"]   = slide.dimensions[1]
        rec["n_levels"] = slide.level_count
        rec["vendor"]   = props.get("openslide.vendor", "")
        rec["mpp_x"]    = float(props.get(openslide.PROPERTY_NAME_MPP_X, 0) or 0)
        rec["mpp_y"]    = float(props.get(openslide.PROPERTY_NAME_MPP_Y, 0) or 0)
        rec["objective_power"] = props.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER,
            props.get("hamamatsu.SourceLens", "")
        )

        # Aperio SVS specific
        rec["model"]     = props.get("aperio.ScanScope ID",
                           props.get("aperio.DSR ID", ""))
        rec["software"]  = props.get("aperio.AppMag",
                           props.get("tiff.Software", ""))
        rec["scan_date"] = props.get("aperio.Date",
                           props.get("aperio.Time", ""))

        # Hamamatsu NDPI specific
        if not rec["model"]:
            rec["model"] = props.get("hamamatsu.ProductVersion",
                           props.get("hamamatsu.InstrumentModel", ""))
        if not rec["software"]:
            rec["software"] = props.get("hamamatsu.Creator", "")
        if not rec["scan_date"]:
            rec["scan_date"] = props.get("hamamatsu.Created", "")

        # print all properties for first slide inspection
        if rec.get("_print_props"):
            print("\nAll properties:")
            for k, v in sorted(props.items()):
                print(f"  {k}: {v}")

        slide.close()

    except Exception as e:
        rec["status"] = "error"
        rec["error"]  = str(e)

    return rec

#  EBRAINS 
print("Extracting EBRAINS scanner metadata...")
ebrains_dir = Path("data/raw/ebrains/wsi")
ndpi_files  = sorted(ebrains_dir.glob("*.ndpi"))
print(f"  Found {len(ndpi_files)} NDPI files")

# print all properties from first slide to find correct keys
if ndpi_files:
    print("\n  Properties from first EBRAINS slide:")
    slide = openslide.OpenSlide(str(ndpi_files[0]))
    for k, v in sorted(slide.properties.items()):
        print(f"    {k}: {v}")
    slide.close()

ebrains_records = []
for i, path in enumerate(ndpi_files):
    rec = extract_scanner_info(path)
    ebrains_records.append(rec)
    if (i+1) % 50 == 0:
        print(f"  Processed {i+1}/{len(ndpi_files)}")

df_ebrains = pd.DataFrame(ebrains_records)
df_ebrains.to_csv(OUT / "ebrains_scanner_metadata.csv", index=False)
print(f"\n  Saved ebrains_scanner_metadata.csv ({len(df_ebrains)} rows)")
print(f"\n  Vendor distribution:")
print(df_ebrains["vendor"].value_counts(dropna=False).to_string())
print(f"\n  Model distribution:")
print(df_ebrains["model"].value_counts(dropna=False).to_string())
print(f"\n  MPP distribution:")
print(df_ebrains["mpp_x"].describe().to_string())
print(f"\n  Objective power:")
print(df_ebrains["objective_power"].value_counts(dropna=False).to_string())

#  TCGA 
print("\nExtracting TCGA scanner metadata...")
tcga_dir  = Path("data/raw/tcga/wsi")
svs_files = sorted(tcga_dir.glob("**/*.svs"))
print(f"  Found {len(svs_files)} SVS files")

# print all properties from first TCGA slide
if svs_files:
    print("\n  Properties from first TCGA slide:")
    slide = openslide.OpenSlide(str(svs_files[0]))
    for k, v in sorted(slide.properties.items()):
        print(f"    {k}: {v}")
    slide.close()

tcga_records = []
for i, path in enumerate(svs_files):
    rec = extract_scanner_info(path)
    # extract case_id from filename
    rec["case_id"] = path.name[:12]
    tcga_records.append(rec)
    if (i+1) % 50 == 0:
        print(f"  Processed {i+1}/{len(svs_files)}")

df_tcga = pd.DataFrame(tcga_records)
df_tcga.to_csv(OUT / "tcga_scanner_metadata.csv", index=False)
print(f"\n  Saved tcga_scanner_metadata.csv ({len(df_tcga)} rows)")
print(f"\n  Vendor distribution:")
print(df_tcga["vendor"].value_counts(dropna=False).to_string())
print(f"\n  Model distribution:")
print(df_tcga["model"].value_counts(dropna=False).to_string())
print(f"\n  MPP distribution:")
print(df_tcga["mpp_x"].describe().to_string())
print(f"\n  Objective power:")
print(df_tcga["objective_power"].value_counts(dropna=False).to_string())

print("\nDone. Files saved to outputs/scanner_metadata/")