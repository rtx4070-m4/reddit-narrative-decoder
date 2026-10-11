from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr

FEATURE_FILE = Path("results/tables/linguistic_features.csv")
VADER_FILE = Path("results/tables/vader_sentiment_results.csv")
EMOTION_FILE = Path("results/tables/transformer_emotion_results.csv")
OUTPUT_DIR = Path("results/tables")

LINGUISTIC_FEATURES = [
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
    "lexical_diversity",
]

EMOTION_PROBABILITIES = [
    "emotion_prob_anger",
    "emotion_prob_disgust",
    "emotion_prob_fear",
    "emotion_prob_joy",
    "emotion_prob_neutral",
    "emotion_prob_sadness",
    "emotion_prob_surprise",
]


def partial_spearman(x, y, controls):
    """Rank-based partial correlation controlling for supplied covariates."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    controls = controls.reset_index(drop=True).copy()

    if len(x) != len(y) or len(x) != len(controls):
        raise ValueError("x, y, and controls must have the same number of rows.")

    z = controls.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if z.shape[1] > 0:
        valid &= np.isfinite(z).all(axis=1)

    x = x[valid]
    y = y[valid]
    z = z[valid, :]

    if len(x) < 10 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return len(x), np.nan, np.nan

    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)

    # Include an intercept and residualize ranked variables against controls.
    design = np.column_stack([np.ones(len(z)), z])
    rx_resid = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ry_resid = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]

    if np.std(rx_resid) == 0 or np.std(ry_resid) == 0:
        return len(x), np.nan, np.nan

    rho, p_value = pearsonr(rx_resid, ry_resid)
    return len(x), float(rho), float(p_value)


def bh_adjust(p_values):
    """Benjamini-Hochberg false-discovery-rate adjustment, preserving NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(p_values))

    if len(valid_indices) == 0:
        return adjusted

    p = p_values[valid_indices]
    order = np.argsort(p)
    sorted_p = p[order]
    m = len(sorted_p)

    sorted_adjusted = sorted_p * m / np.arange(1, m + 1)
    sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]
    sorted_adjusted = np.clip(sorted_adjusted, 0, 1)

    restored = np.empty(m, dtype=float)
    restored[order] = sorted_adjusted
    adjusted[valid_indices] = restored
    return adjusted


def require_columns(frame, required, label):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} file is missing required column(s): {', '.join(missing)}\n"
            f"Available columns: {frame.columns.tolist()}"
        )


def main():
    print("=" * 72)
    print("PHASE 6D — ROBUSTNESS CHECKS: TEXT LENGTH AND COMMUNITY")
    print("=" * 72)

    for path in (FEATURE_FILE, VADER_FILE, EMOTION_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    ling = pd.read_csv(FEATURE_FILE, low_memory=False)
    vader = pd.read_csv(VADER_FILE, low_memory=False)
    emo = pd.read_csv(EMOTION_FILE, low_memory=False)

    for label, frame in (
        ("Linguistic features", ling),
        ("VADER results", vader),
        ("Transformer emotion", emo),
    ):
        require_columns(frame, ["record_id"], label)
        if frame["record_id"].duplicated().any():
            duplicate_count = int(frame["record_id"].duplicated().sum())
            raise ValueError(
                f"{label} contains {duplicate_count} duplicate record_id value(s)."
            )

    require_columns(
        ling,
        ["record_id", "word_count", "community", "year"] + LINGUISTIC_FEATURES,
        "Linguistic features",
    )
    require_columns(vader, ["record_id", "vader_compound"], "VADER results")
    require_columns(emo, ["record_id"] + EMOTION_PROBABILITIES, "Transformer emotion")

    # Merge the VADER score from the VADER file and emotion probabilities
    # from the transformer file. Do not request vader_compound from emo.
    df = ling.merge(
        vader[["record_id", "vader_compound"]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )

    emotion_cols = ["record_id"] + EMOTION_PROBABILITIES
    df = df.merge(
        emo[emotion_cols],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(ling):
        raise ValueError(
            f"Only {len(df):,} of {len(ling):,} linguistic records matched both "
            "the VADER and transformer files. Check record_id consistency."
        )

    print(f"\nMatched observations: {len(df):,}")
    print(f"Communities: {', '.join(sorted(df['community'].dropna().astype(str).unique()))}")

    # Analyze the full corpus and a 2016-only sensitivity subset.
    periods = [("all_years", df.copy())]
    year_numeric = pd.to_numeric(df["year"], errors="coerce")
    subset_2016 = df.loc[year_numeric == 2016].copy()
    if not subset_2016.empty:
        periods.append(("2016_only", subset_2016))
    else:
        print("WARNING: No rows with year == 2016; skipping 2016-only analysis.")

    targets = ["vader_compound"] + EMOTION_PROBABILITIES
    output_rows = []

    for period, subset in periods:
        subset = subset.reset_index(drop=True)

        # Control for log text length and community fixed effects.
        log_word_count = np.log1p(
            pd.to_numeric(subset["word_count"], errors="coerce").clip(lower=0)
        ).rename("log_word_count")

        community_dummies = pd.get_dummies(
            subset["community"].astype("string"),
            prefix="community",
            drop_first=True,
            dtype=float,
        ).reset_index(drop=True)

        controls = pd.concat(
            [log_word_count.reset_index(drop=True), community_dummies],
            axis=1,
        )

        print(f"\nCalculating {period}: {len(subset):,} observations")

        for feature in LINGUISTIC_FEATURES:
            for target in targets:
                pair = subset[[feature, target]].apply(
                    pd.to_numeric, errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)

                valid = pair.notna().all(axis=1)
                if valid.sum() < 10:
                    continue

                raw_rho = pair.loc[valid, feature].corr(
                    pair.loc[valid, target], method="spearman"
                )

                valid_controls = controls.loc[valid].reset_index(drop=True)
                n, adjusted_rho, p_value = partial_spearman(
                    pair.loc[valid, feature].reset_index(drop=True),
                    pair.loc[valid, target].reset_index(drop=True),
                    valid_controls,
                )

                output_rows.append(
                    {
                        "period": period,
                        "feature": feature,
                        "target": target,
                        "n": n,
                        "raw_spearman_rho": float(raw_rho),
                        "adjusted_partial_spearman_rho": adjusted_rho,
                        "adjusted_p_raw": p_value,
                    }
                )

    results = pd.DataFrame(output_rows)
    if results.empty:
        raise RuntimeError("No correlations were calculated; check the input columns.")

    results["adjusted_p_bh_fdr"] = np.nan
    for _, group_indices in results.groupby("period").groups.items():
        results.loc[group_indices, "adjusted_p_bh_fdr"] = bh_adjust(
            results.loc[group_indices, "adjusted_p_raw"].to_numpy()
        )

    results["significant_fdr_0_05"] = results["adjusted_p_bh_fdr"] < 0.05
    results["rho_change"] = (
        results["adjusted_partial_spearman_rho"] - results["raw_spearman_rho"]
    )
    results["absolute_adjusted_rho"] = (
        results["adjusted_partial_spearman_rho"].abs()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "linguistic_sentiment_emotion_adjusted_correlations.csv"
    results.to_csv(output_path, index=False)

    all_years = results.loc[results["period"] == "all_years"].copy()
    print("\nTop adjusted associations by absolute effect size (all years):")
    if not all_years.empty:
        print(
            all_years.sort_values("absolute_adjusted_rho", ascending=False)[
                [
                    "feature",
                    "target",
                    "raw_spearman_rho",
                    "adjusted_partial_spearman_rho",
                    "adjusted_p_bh_fdr",
                    "significant_fdr_0_05",
                ]
            ]
            .head(25)
            .round(6)
            .to_string(index=False)
        )

    print("\nRisk/threat language versus predicted fear:")
    focus = results.loc[
        (results["feature"] == "risk_threat_ratio")
        & (results["target"] == "emotion_prob_fear")
    ]
    if focus.empty:
        print("No matching risk_threat_ratio / emotion_prob_fear result was produced.")
    else:
        print(
            focus[
                [
                    "period",
                    "n",
                    "raw_spearman_rho",
                    "adjusted_partial_spearman_rho",
                    "adjusted_p_raw",
                    "adjusted_p_bh_fdr",
                    "significant_fdr_0_05",
                ]
            ]
            .round(6)
            .to_string(index=False)
        )

    print("\nLargest changes after adjustment (all years):")
    if not all_years.empty:
        print(
            all_years.assign(abs_change=all_years["rho_change"].abs()).sort_values(
                "abs_change", ascending=False
            )[
                [
                    "feature",
                    "target",
                    "raw_spearman_rho",
                    "adjusted_partial_spearman_rho",
                    "rho_change",
                    "adjusted_p_bh_fdr",
                ]
            ]
            .head(15)
            .round(6)
            .to_string(index=False)
        )

    print(f"\nSaved: {output_path}")
    print("\nInterpretation cautions:")
    print("- Estimates adjust statistically for log word count and community.")
    print("- This sensitivity analysis does not establish causality or remove all confounding.")
    print("- Linguistic ratios share a word-count denominator; interpret cautiously.")
    print("- Emotion probabilities are model outputs, not ground-truth labels.")
    print("- The 2016-only subset is a temporal sensitivity check.")
    print("- Input files and previous result files are not modified.")


if __name__ == "__main__":
    main()
