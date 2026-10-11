from pathlib import Path
import numpy as np
import pandas as pd

base = Path("results/tables/phase8_cross_community")
files = [
    "phase8_pairwise_2016_02.csv",
    "phase8_pairwise_2016_only.csv",
    "phase8_pairwise_all_years.csv",
]

for filename in files:
    df = pd.read_csv(base / filename)
    expected = 2 * df["mannwhitney_u"] / (df["n_a"] * df["n_b"]) - 1
    difference = (df["rank_biserial_a_vs_b"] - expected).abs()
    mismatches = int((difference > 1e-8).sum())

    print(f"\nFile: {filename}")
    print(f"Rows checked: {len(df)}")
    print(f"Effect-size mismatches: {mismatches}")
    print(f"Maximum absolute difference: {difference.max():.12g}")
    print("PASS" if mismatches == 0 else "CHECK REQUIRED")
