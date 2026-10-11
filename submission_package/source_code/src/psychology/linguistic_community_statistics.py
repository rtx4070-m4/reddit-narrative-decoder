from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, rankdata


INPUT_FILE = Path("results/tables/linguistic_features.csv")
OUTPUT_DIR = Path("results/tables")

PRIMARY_FEATURES = [
    "word_count",
    "lexical_diversity",
    "first_person_ratio",
    "second_person_ratio",
    "third_person_ratio",
    "negation_ratio",
    "cognitive_ratio",
    "uncertainty_ratio",
    "social_ratio",
    "achievement_ratio",
    "risk_threat_ratio",
    "intensifier_ratio",
    "hedge_ratio",
    "question_word_ratio",
    "first_person_share",
    "second_person_share",
    "third_person_share",
]


def holm_adjust(p_values):
    """Holm family-wise error adjustment; returns values in original order."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return np.array([])
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.maximum.accumulate(
        np.minimum(1.0, (m - np.arange(m)) * sorted_p)
    )
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def epsilon_squared_kw(h_stat, n, k):
    """Common epsilon-squared effect-size estimate for Kruskal-Wallis."""
    if n <= k:
        return np.nan
    return max(0.0, (h_stat - k + 1) / (n - k))


def rank_biserial_u(x, y, u_stat):
    """Rank-biserial effect size for Mann-Whitney U, oriented x versus y."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    # scipy's U statistic corresponds to x; orientation is x relative to y.
    return (2.0 * u_stat) / (nx * ny) - 1.0


def run_analysis(df, features, period_label):
    communities = sorted(df["community"].dropna().unique())
    rows = []
    p_values = []

    for feature in features:
        groups = [
            df.loc[df["community"] == community, feature].dropna().astype(float).to_numpy()
            for community in communities
        ]
        nonempty = [(c, g) for c, g in zip(communities, groups) if len(g) > 0]
        if len(nonempty) < 2:
            continue

        stat, p = kruskal(*[g for _, g in nonempty])
        n = sum(len(g) for _, g in nonempty)
        k = len(nonempty)
        row = {
            "period": period_label,
            "feature": feature,
            "n": n,
            "communities_tested": k,
            "kruskal_h": float(stat),
            "df": k - 1,
            "p_raw": float(p),
            "epsilon_squared": epsilon_squared_kw(float(stat), n, k),
        }
        rows.append(row)
        p_values.append(float(p))

    results = pd.DataFrame(rows)
    if not results.empty:
        results["p_holm_across_features"] = holm_adjust(p_values)
        results["significant_holm_0_05"] = results["p_holm_across_features"] < 0.05
        results = results.sort_values("p_holm_across_features").reset_index(drop=True)

    # Pairwise tests are run for every feature; adjusted p-values are corrected
    # within each feature's six community comparisons. These are exploratory
    # follow-ups; the across-feature omnibus adjustment remains the main screen.
    pair_rows = []
    for feature in features:
        feature_pairs = []
        for community_a, community_b in combinations(communities, 2):
            x = df.loc[df["community"] == community_a, feature].dropna().astype(float).to_numpy()
            y = df.loc[df["community"] == community_b, feature].dropna().astype(float).to_numpy()
            if len(x) == 0 or len(y) == 0:
                continue
            u_stat, p = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
            feature_pairs.append({
                "period": period_label,
                "feature": feature,
                "community_a": community_a,
                "community_b": community_b,
                "n_a": len(x),
                "n_b": len(y),
                "median_a": float(np.median(x)),
                "median_b": float(np.median(y)),
                "mean_a": float(np.mean(x)),
                "mean_b": float(np.mean(y)),
                "mannwhitney_u": float(u_stat),
                "p_raw": float(p),
                "rank_biserial_a_vs_b": float(rank_biserial_u(x, y, u_stat)),
            })
        if feature_pairs:
            adjusted = holm_adjust([r["p_raw"] for r in feature_pairs])
            for record, p_adj in zip(feature_pairs, adjusted):
                record["p_holm_within_feature"] = float(p_adj)
                record["significant_holm_0_05"] = bool(p_adj < 0.05)
                pair_rows.append(record)

    return results, pd.DataFrame(pair_rows)


def make_descriptives(df, features, period_label):
    rows = []
    for community, group in df.groupby("community", dropna=False):
        for feature in features:
            values = group[feature].dropna().astype(float)
            if values.empty:
                continue
            rows.append({
                "period": period_label,
                "community": community,
                "feature": feature,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "median": float(values.median()),
                "q1": float(values.quantile(0.25)),
                "q3": float(values.quantile(0.75)),
                "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("PHASE 6B — COMMUNITY COMPARISON OF LINGUISTIC FEATURES")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run Phase 6A first and execute this "
            "script from the project root."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    required = {"community", "year", *PRIMARY_FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    print(f"\nInput observations: {len(df):,}")
    print(f"Communities: {df['community'].nunique()}")
    print(f"Primary features tested: {len(PRIMARY_FEATURES)}")

    # Overall analysis
    overall_kw, overall_pairwise = run_analysis(df, PRIMARY_FEATURES, "all_years")
    overall_desc = make_descriptives(df, PRIMARY_FEATURES, "all_years")

    # Comparable-period robustness analysis: the corpus is heavily concentrated in 2016.
    df_2016 = df.loc[pd.to_numeric(df["year"], errors="coerce") == 2016].copy()
    if len(df_2016) > 0 and df_2016["community"].nunique() >= 2:
        kw_2016, pairwise_2016 = run_analysis(df_2016, PRIMARY_FEATURES, "2016_only")
        desc_2016 = make_descriptives(df_2016, PRIMARY_FEATURES, "2016_only")
    else:
        kw_2016 = pd.DataFrame()
        pairwise_2016 = pd.DataFrame()
        desc_2016 = pd.DataFrame()

    overall_kw.to_csv(OUTPUT_DIR / "linguistic_community_kruskal_all_years.csv", index=False)
    overall_pairwise.to_csv(OUTPUT_DIR / "linguistic_community_pairwise_all_years.csv", index=False)
    overall_desc.to_csv(OUTPUT_DIR / "linguistic_community_descriptives_all_years.csv", index=False)

    if not kw_2016.empty:
        kw_2016.to_csv(OUTPUT_DIR / "linguistic_community_kruskal_2016.csv", index=False)
        pairwise_2016.to_csv(OUTPUT_DIR / "linguistic_community_pairwise_2016.csv", index=False)
        desc_2016.to_csv(OUTPUT_DIR / "linguistic_community_descriptives_2016.csv", index=False)

    print("\n" + "=" * 72)
    print("OVERALL COMMUNITY DIFFERENCES — ALL YEARS")
    print("=" * 72)
    if not overall_kw.empty:
        print(overall_kw[
            ["feature", "kruskal_h", "p_raw", "p_holm_across_features",
             "epsilon_squared", "significant_holm_0_05"]
        ].round(6).to_string(index=False))

    print("\n" + "=" * 72)
    print("2016-ONLY ROBUSTNESS ANALYSIS")
    print("=" * 72)
    if not kw_2016.empty:
        print(kw_2016[
            ["feature", "kruskal_h", "p_raw", "p_holm_across_features",
             "epsilon_squared", "significant_holm_0_05"]
        ].round(6).to_string(index=False))
    else:
        print("2016-only analysis not available.")

    print("\n" + "=" * 72)
    print("PAIRWISE COMPARISONS")
    print("Showing the 20 smallest adjusted p-values (all years).")
    print("=" * 72)
    if not overall_pairwise.empty:
        print(
            overall_pairwise.sort_values("p_holm_within_feature")[
                ["feature", "community_a", "community_b", "median_a", "median_b",
                 "p_raw", "p_holm_within_feature", "rank_biserial_a_vs_b"]
            ].head(20).round(6).to_string(index=False)
        )

    print("\nSaved outputs in results/tables:")
    print("  linguistic_community_kruskal_all_years.csv")
    print("  linguistic_community_pairwise_all_years.csv")
    print("  linguistic_community_descriptives_all_years.csv")
    if not kw_2016.empty:
        print("  linguistic_community_kruskal_2016.csv")
        print("  linguistic_community_pairwise_2016.csv")
        print("  linguistic_community_descriptives_2016.csv")

    print("\nInterpretation guardrails:")
    print("- Holm-adjusted omnibus p-values account for testing multiple features.")
    print("- Epsilon-squared describes effect size; statistical significance alone is insufficient.")
    print("- Pairwise Mann-Whitney tests are adjusted within each feature.")
    print("- Lexicon ratios describe text, not the psychology or mental health of individual users.")
    print("- Treat 2016-only results as the main robustness check due to uneven year coverage.")
    print("- No source corpus or Phase 6A feature file was modified.")


if __name__ == "__main__":
    main()
