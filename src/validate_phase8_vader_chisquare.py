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

    table = pd.crosstab(subset["community"], subset["vader_label"])
    chi2, p, df, expected = chi2_contingency(table, correction=False)
    n = int(table.to_numpy().sum())
    r, c = table.shape
    v = np.sqrt(chi2 / (n * min(r - 1, c - 1)))

    print(f"\nPeriod: {period}")
    print(f"Records: {n}")
    print(f"Contingency table shape: {table.shape}")
    print(f"Chi-square: {chi2:.10g}")
    print(f"Degrees of freedom: {df}")
    print(f"P-value: {p:.10g}")
    print(f"Cramer's V: {v:.10g}")
    print("Observed counts:")
    print(table.to_string())
