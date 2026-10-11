from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

root = Path(".")
base = root / "results/tables/phase8_cross_community"
out = root / "results/tables/phase12a_statistical_audit"
out.mkdir(parents=True, exist_ok=True)

data_path = base / "phase8_merged_observations.csv"
results_path = base / "phase8_pairwise_all_years.csv"

data = pd.read_csv(data_path)
results = pd.read_csv(results_path)

def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    order = np.argsort(values)
    sorted_p = values[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (n - np.arange(n)) * sorted_p)
    )
    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted
    return adjusted

rows = []

for _, r in results.iterrows():
    feature = str(r["feature"])
    a = str(r["community_a"])
    b = str(r["community_b"])

    item = {
        "feature": feature,
        "community_a": a,
        "community_b": b,
        "reported_n_a": r["n_a"],
        "reported_n_b": r["n_b"],
        "reported_u": r["mannwhitney_u"],
        "recomputed_u": np.nan,
        "reported_p": r["p_raw"],
        "recomputed_p": np.nan,
        "reported_holm": r["p_holm_within_feature"],
        "recomputed_holm": np.nan,
        "status": "REVIEW",
        "detail": "",
    }

    if feature not in data.columns or "community" not in data.columns:
        item["detail"] = "Required source column missing"
        rows.append(item)
        continue

    temp = data[["community", feature]].copy()
    temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
    temp = temp.dropna(subset=["community", feature])

    x = temp.loc[temp["community"].astype(str) == a, feature].to_numpy()
    y = temp.loc[temp["community"].astype(str) == b, feature].to_numpy()

    if len(x) == 0 or len(y) == 0:
        item["detail"] = f"Empty comparison group: n_a={len(x)}, n_b={len(y)}"
        rows.append(item)
        continue

    try:
        test = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        item["recomputed_u"] = test.statistic
        item["recomputed_p"] = test.pvalue
        item["recomputed_n_a"] = len(x)
        item["recomputed_n_b"] = len(y)

        u_ok = abs(float(r["mannwhitney_u"]) - test.statistic) < 1e-7
        n_ok = (
            len(x) == int(r["n_a"])
            and len(y) == int(r["n_b"])
        )
        p_old = float(r["p_raw"])
        p_ok = (
            abs(p_old - test.pvalue) < 1e-10
            or (
                p_old > 0 and test.pvalue > 0
                and abs(np.log10(p_old) - np.log10(test.pvalue)) < 1e-6
            )
        )

        item["status"] = "PASS" if u_ok and n_ok and p_ok else "REVIEW"
        item["detail"] = (
            f"n_a={len(x)}; n_b={len(y)}; "
            f"U_match={u_ok}; counts_match={n_ok}; p_match={p_ok}"
        )
    except Exception as exc:
        item["detail"] = repr(exc)

    rows.append(item)

audit = pd.DataFrame(rows)

# Recalculate Holm adjustments independently within each feature.
audit["recomputed_holm"] = np.nan
for feature, indices in audit.groupby("feature").groups.items():
    idx = list(indices)
    p = pd.to_numeric(audit.loc[idx, "recomputed_p"], errors="coerce")
    if p.notna().all():
        audit.loc[idx, "recomputed_holm"] = holm_adjust(p.to_numpy())

audit["holm_match"] = (
    pd.to_numeric(audit["reported_holm"], errors="coerce")
    - pd.to_numeric(audit["recomputed_holm"], errors="coerce")
).abs() < 1e-8

audit.loc[~audit["holm_match"], "status"] = "REVIEW"
audit.loc[~audit["holm_match"], "detail"] += "; Holm mismatch"

output = out / "phase12c_pairwise_reproduction.csv"
audit.to_csv(output, index=False)

print("\n=== PHASE 12C PAIRWISE REPRODUCTION ===")
print("Total comparisons:", len(audit))
print("\nTest result status:")
print(audit["status"].value_counts(dropna=False).to_string())
print("\nHolm correction matches:", int(audit["holm_match"].sum()), "/", len(audit))

issues = audit[
    (audit["status"] != "PASS") | (~audit["holm_match"])
]
if len(issues):
    print("\nComparisons needing review:")
    print(issues[[
        "feature", "community_a", "community_b",
        "reported_u", "recomputed_u",
        "reported_p", "recomputed_p",
        "reported_holm", "recomputed_holm", "detail"
    ]].to_string(index=False))
else:
    print("\nNo discrepancies detected.")

print("\nSaved:", output.resolve())
print("This verifies numerical reproduction, not all test assumptions.")
