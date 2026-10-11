from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

base = Path("results/tables/phase8_cross_community")
data = pd.read_csv(base / "phase8_merged_observations.csv")

periods = {
    "2016_02": "is_february_2016",
    "2016_only": "is_2016",
    "all_years": None,
}

for period, flag in periods.items():
    subset = data.copy()

    if flag is not None:
        subset = subset[subset[flag].astype(str).str.lower().eq("true")]

    table = pd.crosstab(subset["community"], subset["emotion_label"])
    chi2, p, df, expected = chi2_contingency(table, correction=False)
    n = int(table.to_numpy().sum())
    r, c = table.shape
    v = np.sqrt(chi2 / (n * min(r - 1, c - 1)))

    saved = pd.read_csv(base / f"phase8_emotion_label_chisquare_{period}.csv").iloc[0]

    print(f"\nPeriod: {period}")
    print(f"Records: {n} (reported: {int(saved['n'])})")
    print(f"Chi-square: {chi2:.12g} (reported: {float(saved['chi2']):.12g})")
    print(f"Degrees of freedom: {df} (reported: {int(saved['df'])})")
    print(f"P-value: {p:.12g} (reported: {float(saved['p_value']):.12g})")
    print(f"Cramer's V: {v:.12g} (reported: {float(saved['cramers_v']):.12g})")

    passed = (
        n == int(saved["n"])
        and df == int(saved["df"])
        and np.isclose(chi2, float(saved["chi2"]), rtol=1e-8, atol=1e-8)
        and np.isclose(p, float(saved["p_value"]), rtol=1e-6, atol=1e-15)
        and np.isclose(v, float(saved["cramers_v"]), rtol=1e-8, atol=1e-8)
    )
    print("PASS" if passed else "CHECK REQUIRED")
