import pandas as pd

df = pd.read_csv("tcga_wsi_manifest.txt", sep="\t")

print(f"Before dedup: {len(df)} slides")

# Extract case ID (first 12 chars of filename)
df["case_id"] = df["filename"].str[:12]

# Check which cases have multiple slides
dupes = df[df.duplicated(subset="case_id", keep=False)]
print(f"Cases with multiple slides: {df['case_id'].nunique() - len(df[~df.duplicated(subset='case_id', keep=False)]['case_id'].unique())}")
print(f"\nExample duplicates:")
print(dupes[["filename", "size"]].head(10).to_string())

# Keep one per case ? largest file
df_deduped = df.sort_values("size", ascending=False).drop_duplicates(subset="case_id", keep="first")

print(f"\nAfter dedup: {len(df_deduped)} slides")
print(f"Total size: {df_deduped['size'].sum() / 1e9:.1f} GB")

# Check both projects present
print("\nProject breakdown:")
print(df_deduped["filename"].str.extract(r'(TCGA-GBM|TCGA-LGG)')[0].value_counts())

# Save
df_deduped = df_deduped.drop(columns="case_id")
df_deduped.to_csv("tcga_wsi_manifest_deduped.txt", sep="\t", index=False)
print("\nSaved to tcga_wsi_manifest_deduped.txt")