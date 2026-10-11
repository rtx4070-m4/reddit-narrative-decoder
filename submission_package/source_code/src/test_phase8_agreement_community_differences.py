from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

base = Path("results/tables/phase8_cross_community")
df = pd.read_csv(base / "phase8_merged_observations.csv")

required = {"community", "vader_label", "emotion_label"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# Binary classification: neutral versus non-neutral.
df["vader_binary"] = np.where(
    df["vader_label"].eq("neutral"), "neutral", "non-neutral"
)
df["emotion_binary"] = np.where(
    df["emotion_label"].eq("neutral"), "neutral", "non-neutral"
)
df["agree"] = df["vader_binary"].eq(df["emotion_binary"])

def wilson_interval(successes, n, z=1.959963984540054):
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    half_width = (
        z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
        / denominator
    )
    return center - half_width, center + half_width

def kappa_from_arrays(a, b):
    # Binary labels encoded as 0 and 1.
    po = np.mean(a == b)
    pa = np.mean(a)
    pb = np.mean(b)
    pe = pa * pb + (1 - pa) * (1 - pb)
    if np.isclose(1 - pe, 0):
        return np.nan
    return (po - pe) / (1 - pe)

# Test whether agreement/disagreement proportions vary by community.
contingency = pd.crosstab(df["community"], df["agree"]).reindex(
    columns=[True, False], fill_value=0
)
contingency.columns = ["agreement", "disagreement"]

chi2, p_value, dof, expected = chi2_contingency(contingency)
n_total = contingency.to_numpy().sum()
min_dimension = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
cramers_v = np.sqrt(chi2 / (n_total * min_dimension))

test_result = pd.DataFrame([{
    "test": "Pearson chi-square: agreement vs disagreement by community",
    "n": int(n_total),
    "chi_square": chi2,
    "degrees_of_freedom": dof,
    "p_value": p_value,
    "cramers_v": cramers_v,
    "interpretation_note": (
        "Tests whether observed binary agreement rates vary by community; "
        "does not establish classifier accuracy."
    )
}])

test_path = base / "phase8_agreement_community_chisquare.csv"
test_result.to_csv(test_path, index=False)

# Community estimates and confidence intervals.
rng = np.random.default_rng(20261010)
B = 2000
rows = []

for community, group in df.groupby("community", sort=True):
    n = len(group)
    successes = int(group["agree"].sum())
    agreement = successes / n
    ci_low, ci_high = wilson_interval(successes, n)

    # Encode each classifier's binary labels for bootstrap kappa.
    a = (group["vader_binary"].to_numpy() == "non-neutral").astype(int)
    b = (group["emotion_binary"].to_numpy() == "non-neutral").astype(int)
    kappa = cohen_kappa_score(a, b)

    bootstrap_kappas = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        value = kappa_from_arrays(a[idx], b[idx])
        if np.isfinite(value):
            bootstrap_kappas.append(value)

    if bootstrap_kappas:
        kappa_low, kappa_high = np.percentile(
            bootstrap_kappas, [2.5, 97.5]
        )
    else:
        kappa_low, kappa_high = np.nan, np.nan

    rows.append({
        "community": community,
        "n": n,
        "agreement_count": successes,
        "agreement_percent": agreement * 100,
        "agreement_ci95_low_percent": ci_low * 100,
        "agreement_ci95_high_percent": ci_high * 100,
        "cohen_kappa": kappa,
        "kappa_bootstrap_ci95_low": kappa_low,
        "kappa_bootstrap_ci95_high": kappa_high,
        "bootstrap_replicates_requested": B,
        "bootstrap_replicates_valid": len(bootstrap_kappas)
    })

estimates = pd.DataFrame(rows)
estimates_path = base / "phase8_agreement_community_confidence_intervals.csv"
estimates.to_csv(estimates_path, index=False)

print("\nAGREEMENT-RATE CONTINGENCY TABLE")
print(contingency.to_string())

print("\nOVERALL COMMUNITY DIFFERENCE TEST")
print(test_result.to_string(index=False))

print("\nCOMMUNITY ESTIMATES AND 95% CONFIDENCE INTERVALS")
print(estimates.round(5).to_string(index=False))

print("\nVALIDATION")
print("Community sample sizes sum to:", int(estimates["n"].sum()))
print("Merged data rows:", len(df))
print("Sample-size check:", "PASS" if estimates["n"].sum() == len(df) else "FAIL")

print("\nSaved files:")
print(test_path)
print(estimates_path)
