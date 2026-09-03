import requests
import os
import pandas as pd
from pathlib import Path
from urllib.parse import quote

# -- config --------------------------------------------------------------------
TOKEN     = os.environ.get("EBRAINS_TOKEN")
DATASET   = "8fc108ab-e2b4-406f-8999-60269dc1f994"
BASE_URL  = f"https://data-proxy.ebrains.eu/api/v1/datasets/{DATASET}"
OUT_DIR   = Path("data/raw/ebrains/wsi")
UUID_FILE = Path("manifests/ebrains_uuids.txt")

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not TOKEN:
    raise ValueError("EBRAINS_TOKEN not set")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# -- load target UUIDs ---------------------------------------------------------
with open(UUID_FILE) as f:
    target_uuids = set(line.strip() for line in f if line.strip())
print(f"Target UUIDs: {len(target_uuids)}")

# -- IDH folders to scan -------------------------------------------------------
IDH_FOLDERS = [
    "v1.0/Anaplastic astrocytoma, IDH-mutant/",
    "v1.0/Anaplastic astrocytoma, IDH-wildtype/",
    "v1.0/Anaplastic oligodendroglioma, IDH-mutant and 1p-19q codeleted/",
    "v1.0/Diffuse astrocytoma, IDH-mutant/",
    "v1.0/Diffuse astrocytoma, IDH-wildtype/",
    "v1.0/Glioblastoma, IDH-mutant/",
    "v1.0/Glioblastoma, IDH-wildtype/",
    "v1.0/Oligodendroglioma, IDH-mutant and 1p-19q codeleted/",
    "v1.0/Gemistocytic astrocytoma, IDH-mutant/",
]

# -- list files from all IDH folders ------------------------------------------
def list_folder(folder):
    files = []
    marker = None
    while True:
        params = {"prefix": folder, "limit": 1000}
        if marker:
            params["marker"] = marker
        resp = requests.get(BASE_URL, headers=HEADERS, params=params)
        if resp.status_code != 200:
            print(f"Error listing {folder}: {resp.status_code}")
            break
        data = resp.json()
        batch = data.get("objects", [])
        files.extend(batch)
        # check if there are more pages
        if len(batch) < 1000:
            break
        marker = batch[-1]["name"]
    return files

print("Listing files from IDH folders...")
all_files = []
for folder in IDH_FOLDERS:
    print(f"  Scanning: {folder}")
    files = list_folder(folder)
    ndpi = [f for f in files if f["name"].endswith(".ndpi")]
    print(f"    Found: {len(ndpi)} NDPI files")
    all_files.extend(ndpi)

print(f"Total NDPI files found: {len(all_files)}")

# -- match to target UUIDs -----------------------------------------------------
def get_uuid(file_path):
    return Path(file_path).stem

target_files = [
    f for f in all_files
    if get_uuid(f["name"]) in target_uuids
]
print(f"Matched to cohort: {len(target_files)}")

# check for missing
found_uuids = {get_uuid(f["name"]) for f in target_files}
missing = target_uuids - found_uuids
if missing:
    print(f"WARNING: {len(missing)} UUIDs not found")
    with open("manifests/ebrains_missing_uuids.txt", "w") as f:
        for m in sorted(missing):
            f.write(m + "\n")

# -- get download URL for a file -----------------------------------------------
def get_download_url(file_path):
    encoded = quote(file_path, safe='/')
    resp = requests.get(
        f"{BASE_URL}/{encoded}",
        headers=HEADERS,
        params={"redirect": "false"},
        allow_redirects=False
    )
    if resp.status_code == 302:
        return resp.headers.get("Location")
    elif resp.status_code == 200:
        data = resp.json()
        return data.get("url") or data.get("download_url")
    else:
        # try direct streaming
        return None

# -- download ------------------------------------------------------------------
print(f"\nStarting download to {OUT_DIR}/")
downloaded, skipped, failed = 0, 0, []
total = len(target_files)

for i, f in enumerate(target_files, 1):
    file_path = f["name"]
    filename  = Path(file_path).name
    out_path  = OUT_DIR / filename

    if out_path.exists() and out_path.stat().st_size > 0:
        skipped += 1
        if i % 50 == 0:
            print(f"[{i}/{total}] skipped: {skipped}, downloaded: {downloaded}, failed: {len(failed)}")
        continue

    try:
        # try to get presigned URL first
        dl_url = get_download_url(file_path)

        if dl_url:
            dl_resp = requests.get(dl_url, stream=True, timeout=600)
        else:
            # stream directly from dataset endpoint
            encoded = quote(file_path, safe='/')
            dl_resp = requests.get(
                f"{BASE_URL}/{encoded}",
                headers=HEADERS,
                stream=True,
                timeout=600
            )

        dl_resp.raise_for_status()

        with open(out_path, "wb") as out:
            for chunk in dl_resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                out.write(chunk)

        size_mb = out_path.stat().st_size / 1e6
        downloaded += 1
        print(f"[{i}/{total}] - {filename} ({size_mb:.0f} MB)")

    except Exception as e:
        failed.append((filename, str(e)))
        if out_path.exists():
            out_path.unlink()  # remove partial file
        print(f"[{i}/{total}] - FAILED: {filename} - {e}")

# -- summary -------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Downloaded:    {downloaded}")
print(f"Skipped:       {skipped}")
print(f"Failed:        {len(failed)}")
print(f"Total on disk: {len(list(OUT_DIR.glob('*.ndpi')))}")

if failed:
    with open("manifests/ebrains_failed.txt", "w") as f:
        for name, err in failed:
            f.write(f"{name}\t{err}\n")
    print("Failed list saved to manifests/ebrains_failed.txt")