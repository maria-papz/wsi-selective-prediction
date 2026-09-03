# scripts/check_race_attrition.py
import pandas as pd

# full clinical table before filtering
df_all = pd.read_csv("data/raw/tcga/clinical/tcga_clinical.csv")
df_all["race"] = df_all["demographic.race"].str.lower().fillna("not reported")

# usable manifest
df_use = pd.read_csv("data/raw/tcga/clinical/tcga_manifest.csv")
df_use["race"] = df_use["race"].str.lower().fillna("not reported")

print("Black/AA patients at each stage:")
print(f"  All 1133 GDC cases:        {(df_all['race']=='black or african american').sum()}")

no_idh = df_all[df_all["idh_status"].isna()]
print(f"  Lost to missing IDH:       {(no_idh['race']=='black or african american').sum()}")

no_dx = df_all[df_all["dx_file_id"].isna()]
print(f"  Lost to no DX slide:       {(no_dx['race']=='black or african american').sum()}")

small = df_all[df_all["dx_size_flag"]==True]
print(f"  Lost to small slide <50MB: {(small['race']=='black or african american').sum()}")

print(f"  Usable (manifest):         {(df_use['race']=='black or african american').sum()}")