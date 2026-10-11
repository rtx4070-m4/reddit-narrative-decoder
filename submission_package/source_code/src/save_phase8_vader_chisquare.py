from pathlib import Path
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

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

    table = pd.crosstab(subset["community"], subset["vader_label"])
    chi2, p, df, _ = chi2_contingency(table, correction=False)

    n = int(table.to_numpy().sum())
    rows, cols = table.shape
    cramers_v = np.sqrt(chi2 / (n * min(rows - 1, cols - 1)))

    results.append({
        "period": period,
        "n": n,
        "chi2": chi2,
        "df": df,
        "p_value": p,
        "cramers_v": cramers_v,
    })

output = base / "phase8_vader_label_chisquare.csv"
pd.DataFrame(results).to_csv(output, index=False)

print("Saved:", output)
print(pd.DataFrame(results).to_string(index=False))
