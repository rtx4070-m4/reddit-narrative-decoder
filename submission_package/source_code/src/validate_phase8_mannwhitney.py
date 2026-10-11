from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(".")
TABLES = ROOT / "results" / "tables" / "phase8_cross_community"

merged = pd.read_csv(TABLES / "phase8_merged_observations.csv")
periods = {
    "2016_02": "is_february_2016",
    "2016_only": "is_2016",
    "all_years": None,
}

reports = []

for period, flag in periods.items():
    pair_file = TABLES / f"phase8_pairwise_{period}.csv"
    pairs = pd.read_csv(pair_file)
    data = merged.copy()

    if flag is not None:
        data = data[data[flag].astype(str).str.lower().eq("true")]

    mismatches = []
    skipped = []

    for row in pairs.itertuples(index=False):
        a = data.loc[data["community"].eq(row.community_a), row.feature].dropna()
        b = data.loc[data["community"].eq(row.community_b), row.feature].dropna()

        if len(a) != int(row.n_a) or len(b) != int(row.n_b):
            mismatches.append((row.feature, row.community_a, row.community_b,
                               "sample_size", len(a), len(b), row.n_a, row.n_b))
            continue

        result = mannwhitneyu(a, b, alternative="two-sided", method="auto")
        if not np.isclose(result.statistic, float(row.mannwhitney_u),
                          rtol=1e-8, atol=1e-7):
            mismatches.append((row.feature, row.community_a, row.community_b,
                               "U_statistic", result.statistic, row.mannwhitney_u))

        if not np.isclose(result.pvalue, float(row.p_raw),
                          rtol=1e-6, atol=1e-12):
            mismatches.append((row.feature, row.community_a, row.community_b,
                               "raw_p_value", result.pvalue, row.p_raw))

    reports.append({
        "period": period,
        "source_rows": len(data),
        "pairwise_rows": len(pairs),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:5],
    })

for report in reports:
    print(f"\nPeriod: {report['period']}")
    print(f"Source rows: {report['source_rows']}")
    print(f"Pairwise rows checked: {report['pairwise_rows']}")
    print(f"Mismatches: {report['mismatch_count']}")
    if report["first_mismatches"]:
        print("Examples:")
        for item in report["first_mismatches"]:
            print(" ", item)
    else:
        print("PASS: all sample sizes, U statistics, and raw p-values match.")
