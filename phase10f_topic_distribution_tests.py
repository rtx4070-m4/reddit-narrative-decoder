
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

# ============================================================
# PHASE 10F: STATISTICAL TESTING OF TOPIC DISTRIBUTIONS
# February 2016 only.
# Uses Phase 10E assigned-topic counts.
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    ROOT / "results" / "tables"
    / "phase10e_february_cross_community"
    / "phase10e_february_community_topic_prevalence.csv"
)

OUT = (
    ROOT / "results" / "tables"
    / "phase10f_topic_distribution_tests"
)
OUT.mkdir(parents=True, exist_ok=True)

COMMUNITIES = [
    "gaming", "science", "todayilearned", "worldnews"
]


def holm_adjust(p_values):
    """Holm family-wise error adjustment."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted


def cramers_v(chi2, n, rows, cols):
    """Standard Cramer's V effect size."""
    denominator = n * min(rows - 1, cols - 1)
    if denominator <= 0:
        return np.nan
    return float(np.sqrt(chi2 / denominator))


def main():
    print("=" * 76)
    print("PHASE 10F: TOPIC DISTRIBUTION STATISTICAL TESTS")
    print("=" * 76)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Phase 10E input file not found:\n" + str(INPUT_FILE)
        )

    data = pd.read_csv(INPUT_FILE)

    required = {
        "community", "topic", "topic_records", "model"
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    data["topic_records"] = pd.to_numeric(
        data["topic_records"], errors="raise"
    ).astype(int)

    if (data["topic_records"] < 0).any():
        raise ValueError("Negative topic counts found.")

    models = sorted(data["model"].dropna().unique())
    overall_results = []
    expected_results = []
    residual_results = []
    pairwise_results = []

    for model_name in models:
        print("\n" + "-" * 76)
        print("MODEL:", model_name)

        subset = data[data["model"] == model_name].copy()

        table = subset.pivot_table(
            index="community",
            columns="topic",
            values="topic_records",
            aggfunc="sum",
            fill_value=0,
        )

        # Ensure consistent community order.
        table = table.reindex(
            [c for c in COMMUNITIES if c in table.index],
            fill_value=0,
        )

        # Remove any topic columns that have no assigned records.
        table = table.loc[:, table.sum(axis=0) > 0]

        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError(
                f"Insufficient contingency table dimensions for {model_name}"
            )

        observed = table.to_numpy(dtype=float)
        chi2, p_value, df, expected = chi2_contingency(
            observed, correction=False
        )
        n = observed.sum()
        v = cramers_v(
            chi2, n, observed.shape[0], observed.shape[1]
        )

        expected_min = float(expected.min())
        expected_below5 = int((expected < 5).sum())
        expected_below1 = int((expected < 1).sum())
        cells = int(expected.size)

        overall_results.append({
            "model": model_name,
            "records_assigned": int(n),
            "communities": observed.shape[0],
            "topics": observed.shape[1],
            "chi_square": chi2,
            "degrees_of_freedom": df,
            "p_value": p_value,
            "cramers_v": v,
            "minimum_expected_count": expected_min,
            "expected_cells_below_5": expected_below5,
            "expected_cells_below_1": expected_below1,
            "total_cells": cells,
        })

        print("\nObserved topic counts:")
        print(table.to_string())

        print("\nOverall test:")
        print(f"Assigned records: {int(n)}")
        print(f"Chi-square: {chi2:.4f}")
        print(f"Degrees of freedom: {df}")
        print(f"P-value: {p_value:.8g}")
        print(f"Cramer's V: {v:.4f}")
        print(f"Minimum expected count: {expected_min:.3f}")
        print(
            f"Expected cells below 5: {expected_below5}/{cells}"
        )
        print(
            f"Expected cells below 1: {expected_below1}/{cells}"
        )

        # Expected counts and adjusted standardized residuals.
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        row_props = row_totals / n
        col_props = col_totals / n

        adjusted_residuals = (
            observed - expected
        ) / np.sqrt(
            expected
            * (1 - row_props[:, None])
            * (1 - col_props[None, :])
        )

        raw_cell_p = 2 * norm.sf(np.abs(adjusted_residuals))
        flat_adjusted = holm_adjust(raw_cell_p.ravel())
        adjusted_cell_p = flat_adjusted.reshape(raw_cell_p.shape)

        for i, community in enumerate(table.index):
            for j, topic in enumerate(table.columns):
                expected_results.append({
                    "model": model_name,
                    "community": community,
                    "topic": topic,
                    "observed_count": int(observed[i, j]),
                    "expected_count": expected[i, j],
                })

                residual_results.append({
                    "model": model_name,
                    "community": community,
                    "topic": topic,
                    "observed_count": int(observed[i, j]),
                    "expected_count": expected[i, j],
                    "adjusted_residual": adjusted_residuals[i, j],
                    "cell_p_value": raw_cell_p[i, j],
                    "holm_adjusted_cell_p": adjusted_cell_p[i, j],
                    "observed_above_expected": (
                        observed[i, j] > expected[i, j]
                    ),
                })

        # Pairwise community comparisons within this model.
        pair_records = []
        for community_a, community_b in itertools.combinations(
            table.index.tolist(), 2
        ):
            pair_table = table.loc[[community_a, community_b]]
            pair_observed = pair_table.to_numpy(dtype=float)

            pair_chi2, pair_p, pair_df, pair_expected = (
                chi2_contingency(
                    pair_observed, correction=False
                )
            )

            pair_n = pair_observed.sum()
            pair_v = cramers_v(
                pair_chi2,
                pair_n,
                pair_observed.shape[0],
                pair_observed.shape[1],
            )

            pair_records.append({
                "model": model_name,
                "community_a": community_a,
                "community_b": community_b,
                "records_assigned": int(pair_n),
                "chi_square": pair_chi2,
                "degrees_of_freedom": pair_df,
                "p_value": pair_p,
                "cramers_v": pair_v,
                "minimum_expected_count": float(pair_expected.min()),
                "expected_cells_below_5": int(
                    (pair_expected < 5).sum()
                ),
            })

        if pair_records:
            pair_p_adj = holm_adjust([
                row["p_value"] for row in pair_records
            ])
            for row, adjusted_p in zip(pair_records, pair_p_adj):
                row["holm_adjusted_p"] = adjusted_p
                row["significant_after_holm_0_05"] = (
                    adjusted_p < 0.05
                )
                pairwise_results.append(row)

        print("\nPairwise community tests:")
        if pair_records:
            pair_df = pd.DataFrame(pair_records)
            pair_df["holm_adjusted_p"] = holm_adjust(
                pair_df["p_value"].to_numpy()
            )
            print(
                pair_df[
                    [
                        "community_a", "community_b",
                        "chi_square", "p_value",
                        "holm_adjusted_p", "cramers_v",
                        "minimum_expected_count",
                    ]
                ].to_string(index=False)
            )

    # Save all outputs.
    outputs = [
        (
            pd.DataFrame(overall_results),
            "phase10f_overall_chi_square_results.csv",
        ),
        (
            pd.DataFrame(expected_results),
            "phase10f_expected_cell_counts.csv",
        ),
        (
            pd.DataFrame(residual_results),
            "phase10f_adjusted_residuals.csv",
        ),
        (
            pd.DataFrame(pairwise_results),
            "phase10f_pairwise_community_tests.csv",
        ),
    ]

    for frame, filename in outputs:
        destination = OUT / filename
        frame.to_csv(destination, index=False)
        print("\nSaved:", destination)
        print("Rows:", len(frame))

    print("\n" + "=" * 76)
    print("PHASE 10F COMPLETED")
    print("Output directory:", OUT)
    print("Source files were not modified.")
    print("=" * 76)


if __name__ == "__main__":
    main()
