from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kruskal

base = Path("results/tables/phase8_cross_community")
data = pd.read_csv(base / "phase8_merged_observations.csv")

periods = {
    "2016_02": "is_february_2016",
    "2016_only": "is_2016",
    "all_years": None,
}

results = []

for period, flag in periods.items():
    subset = data.copy()

    if flag is not None:
        subset = subset[subset[flag].astype(str).str.lower().eq("true")]

    saved = pd.read_csv(base / f"phase8_kruskal_{period}.csv")
    h_mismatches = 0
    effect_mismatches = 0
    n_mismatches = 0

    for row in saved.itertuples(index=False):
        groups = [
            subset.loc[subset["community"].eq(c), row.feature].dropna().to_numpy()
            for c in sorted(subset["community"].dropna().unique())
        ]

        if any(len(g) == 0 for g in groups):
            print(f"Empty group: {period}, {row.feature}")
            continue

        n = sum(len(g) for g in groups)
        test = kruskal(*groups)
        expected_eps = max(0.0, (test.statistic - len(groups) + 1) / (n - len(groups)))

        if n != int(row.n):
            n_mismatches += 1

        if not np.isclose(test.statistic, float(row.kruskal_h), rtol=1e-8, atol=1e-8):
            h_mismatches += 1

        if not np.isclose(expected_eps, float(row.epsilon_squared), rtol=1e-7, atol=1e-9):
            effect_mismatches += 1

    results.append({
        "period": period,
        "source_rows": len(subset),
        "features": len(saved),
        "H_mismatches": h_mismatches,
        "epsilon_squared_mismatches": effect_mismatches,
        "sample_size_mismatches": n_mismatches,
    })

for r in results:
    print(f"\nPeriod: {r['period']}")
    print(f"Source rows: {r['source_rows']}")
    print(f"Features checked: {r['features']}")
    print(f"H-statistic mismatches: {r['H_mismatches']}")
    print(f"Epsilon-squared mismatches: {r['epsilon_squared_mismatches']}")
    print(f"Sample-size mismatches: {r['sample_size_mismatches']}")

    passed = (
        r["features"] == 25
        and r["H_mismatches"] == 0
        and r["epsilon_squared_mismatches"] == 0
        and r["sample_size_mismatches"] == 0
    )
    print("PASS" if passed else "CHECK REQUIRED")
