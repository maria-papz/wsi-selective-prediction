# scripts/race_stats.py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from scipy.stats import norm

df = pd.read_csv("data/raw/tcga/clinical/tcga_manifest.csv")
df["race"] = df["race"].str.lower()

known = df[~df["race"].isin(["not reported", "unknown", "nan"])].copy()
known["race_binary"] = known["race"].apply(
    lambda x: "White" if x == "white" else "Other"
)

# contingency table
ct = pd.crosstab(known["race_binary"], known["idh_status"])
print("Contingency table:")
print(ct.to_string())

# chi-square
chi2, p_chi2, dof, expected = chi2_contingency(ct)
print(f"\nChi-square = {chi2:.3f}, p = {p_chi2:.4f}")

# odds ratio + 95% CI
a = ct.loc["Other",  "mutant"]
b = ct.loc["Other",  "wildtype"]
c = ct.loc["White",  "mutant"]
d = ct.loc["White",  "wildtype"]

or_val = (a * d) / (b * c)
log_or = np.log(or_val)
se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
ci_lo = np.exp(log_or - 1.96 * se_log_or)
ci_hi = np.exp(log_or + 1.96 * se_log_or)

print(f"\nOdds ratio (Other vs White): {or_val:.3f} (95% CI {ci_lo:.3f}-{ci_hi:.3f})")
print(f"Interpretation: Other-race patients have {or_val:.2f}x the odds of IDH-mutant")

# absolute risk difference + 95% CI
p_other = a / (a + b)
p_white = c / (c + d)
rd      = p_other - p_white
se_rd   = np.sqrt(p_other*(1-p_other)/(a+b) + p_white*(1-p_white)/(c+d))
rd_lo   = rd - 1.96 * se_rd
rd_hi   = rd + 1.96 * se_rd

print(f"\nRisk difference (Other - White): {100*rd:.1f}pp "
      f"(95% CI {100*rd_lo:.1f}-{100*rd_hi:.1f} pp)")

# Cramer's V (effect size)
n   = len(known)
v   = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print(f"\nCramér's V = {v:.3f}  (effect size: small-0.1, medium-0.3)")

# breakdown by project
print("\n--- Stratified by project ---")
for proj in ["TCGA-GBM", "TCGA-LGG"]:
    sub = known[known["project"] == proj]
    ct_p = pd.crosstab(sub["race_binary"], sub["idh_status"])
    print(f"\n{proj}:")
    print(ct_p.to_string())
    if ct_p.shape == (2, 2):
        _, p_p = fisher_exact(ct_p)
        for grp in ["White", "Other"]:
            if grp in ct_p.index:
                s  = ct_p.loc[grp]
                total = s.sum()
                pct = 100 * s.get("mutant", 0) / total
                print(f"  {grp}: {pct:.1f}% IDH-mutant (n={total})")
        print(f"  Fisher's exact p = {p_p:.4f}")

# sex comparison
print("\n--- Sex comparison ---")
ct_sex = pd.crosstab(df["sex"].str.lower(), df["idh_status"])
ct_sex = ct_sex.loc[ct_sex.index.isin(["male","female"])]
print(ct_sex.to_string())
chi2_s, p_sex, _, _ = chi2_contingency(ct_sex)
for sx in ["male","female"]:
    if sx in ct_sex.index:
        s   = ct_sex.loc[sx]
        pct = 100 * s.get("mutant",0) / s.sum()
        print(f"  {sx.capitalize()}: {pct:.1f}% IDH-mutant (n={s.sum()})")
print(f"Chi-square = {chi2_s:.3f}, p = {p_sex:.4f}")

# age comparison
print("\n--- Age comparison (mutant vs wildtype) ---")
from scipy.stats import mannwhitneyu
mut_age = df[df["idh_status"]=="mutant"]["age"].dropna()
wt_age  = df[df["idh_status"]=="wildtype"]["age"].dropna()
stat, p_age = mannwhitneyu(mut_age, wt_age, alternative="two-sided")
print(f"  IDH-mutant:   median {mut_age.median():.0f}y "
      f"(IQR {mut_age.quantile(0.25):.0f}-{mut_age.quantile(0.75):.0f})")
print(f"  IDH-wildtype: median {wt_age.median():.0f}y "
      f"(IQR {wt_age.quantile(0.25):.0f}-{wt_age.quantile(0.75):.0f})")
print(f"  Mann-Whitney U p = {p_age:.2e}")