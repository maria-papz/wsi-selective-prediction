import requests
import pandas as pd
import json
from pathlib import Path

# load file IDs
df = pd.read_csv("manifests/tcga_gdc_manifest.txt")
print(f"File IDs: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(df.head(3))

# fetch proper manifest from GDC API
file_ids = df["file_id"].tolist()
print(f"\nFetching GDC manifest for {len(file_ids)} files...")

r = requests.post(
    "https://api.gdc.cancer.gov/manifest",
    headers={"Content-Type": "application/json"},
    data=json.dumps({"ids": file_ids})
)
print(f"Status: {r.status_code}")
print(f"Response preview:\n{r.text[:300]}")

if r.status_code == 200:
    with open("manifests/tcga_gdc_manifest_fixed.txt", "w") as f:
        f.write(r.text)
    print(f"\nSaved manifests/tcga_gdc_manifest_fixed.txt")
    print(f"Lines: {len(r.text.strip().split(chr(10)))}")