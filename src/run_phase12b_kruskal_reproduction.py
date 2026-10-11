from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal

root = Path(".")
base = root / "results/tables/phase8_cross_community"
data_path = base / "phase8_merged_observations.csv"
results_path = base / "phase8_kruskal_all_years.csv"
out = root / "results/tables/phase12a_statistical_audit"
out.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
    raise FileNotFoundError(f"Missing underlying data: {data_path}")
if not results_path.exists():
    raise FileNotFoundError(f"Missing results table: {results_path}")

data = pd.read_csv(data_path)
results = pd.read_csv(results_path)

if "community" not in data.columns:
    raise ValueError("Underlying data has no 'community' column.")

groups_found = sorted(data["community"].dropna().astype(str).unique())
print("Communities found:", groups_found)
print("Underlying rows:", len(data))
print("Reported test rows:", len(results))

rows = []

for _, result in results.iterrows():
    feature = str(result["feature"])
    row = {
        "period": result.get("period", ""),
        "feature": feature,
        "reported_n": result.get("n", np.nan),
        "recomputed_n": np.nan,
        "reported_h": result.get("kruskal_h", np.nan),
        "recomputed_h": np.nan,
        "reported_p": result.get("p_raw", np.nan),
        "recomputed_p": np.nan,
        "reported_epsilon_squared": result.get("epsilon_squared", np.nan),
        "recomputed_epsilon_squared": np.nan,
        "h_absolute_difference": np.nan,
        "p_absolute_difference": np.nan,
        "status": "",
        "detail": "",
    }

    if feature not in data.columns:
        row["status"] = "REVIEW"
        row["detail"] = "Feature absent from underlying data"
        rows.append(row)
        continue

    numeric = pd.to_numeric(data[feature], errors="coerce")
    valid_data = data.loc[numeric.notna(), ["community"]].copy()
    valid_data["_value"] = numeric[numeric.notna()]

    groups = [
        g["_value"].to_numpy()
        for _, g in valid_data.groupby("community")
        if len(g) > 0
    ]

    n = sum(len(g) for g in groups)
    k = len(groups)

    row["recomputed_n"] = n

    if k < 2 or n <= k:
        row["status"] = "REVIEW"
        row["detail"] = f"Insufficient valid groups: k={k}, n={n}"
        rows.append(row)
        continue

    try:
        h_stat, p_value = kruskal(*groups)
        epsilon = max(0.0, (h_stat - k + 1) / (n - k))

        row["recomputed_h"] = h_stat
        row["recomputed_p"] = p_value
        row["recomputed_epsilon_squared"] = epsilon

        reported_h = pd.to_numeric(
            pd.Series([row["reported_h"]]), errors="coerce"
        ).iloc[0]
        reported_p = pd.to_numeric(
            pd.Series([row["reported_p"]]), errors="coerce"
        ).iloc[0]
        reported_eps = pd.to_numeric(
            pd.Series([row["reported_epsilon_squared"]]), errors="coerce"
        ).iloc[0]

        h_diff = abs(h_stat - reported_h) if pd.notna(reported_h) else np.nan
        p_diff = abs(p_value - reported_p) if pd.notna(reported_p) else np.nan
        row["h_absolute_difference"] = h_diff
        row["p_absolute_difference"] = p_diff

        checks = [
            n == int(float(row["reported_n"])),
            pd.notna(h_diff) and h_diff < 1e-6,
            pd.notna(p_diff) and (
                p_diff < 1e-10 or (
                    p_value > 0 and reported_p > 0
                    and abs(np.log10(p_value) - np.log10(reported_p)) < 1e-6
                )
            ),
            pd.notna(reported_eps) and abs(epsilon - reported_eps) < 1e-6,
        ]

        row["status"] = "PASS" if all(checks) else "REVIEW"
        row["detail"] = (
            f"groups={k}; valid_n={n}; "
            f"H_diff={h_diff}; p_diff={p_diff}; "
            f"epsilon_diff={abs(epsilon-reported_eps) if pd.notna(reported_eps) else 'NA'}"
        )

    except Exception as exc:
        row["status"] = "REVIEW"
        row["detail"] = repr(exc)

    rows.append(row)

audit = pd.DataFrame(rows)
audit.to_csv(out / "phase12b_kruskal_reproduction.csv", index=False)

print("\n=== PHASE 12B: KRUSKAL-WALLIS REPRODUCTION ===")
print(audit[["feature", "reported_n", "recomputed_n",
             "reported_h", "recomputed_h",
             "reported_p", "recomputed_p",
             "status", "detail"]].to_string(index=False))

print("\nStatus counts:")
print(audit["status"].value_counts(dropna=False).to_string())
print("\nSaved:", (out / "phase12b_kruskal_reproduction.csv").resolve())
print("\nA REVIEW result means investigate the discrepancy; it does not automatically mean the original result is wrong.")
