from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


FEATURE_FILE = Path("results/tables/linguistic_features.csv")
VADER_FILE = Path("results/tables/vader_sentiment_results.csv")
EMOTION_FILE = Path("results/tables/transformer_emotion_results.csv")
OUTPUT_DIR = Path("results/tables")

LINGUISTIC_FEATURES = [
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
]

EMOTION_NAMES = {
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
}


def find_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    return None


def bh_adjust(p_values):
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def identify_emotion_probability_columns(df):
    cols = []
    for col in df.columns:
        name = str(col).lower()
        if any(excluded in name for excluded in ("label", "confidence", "score", "predicted")):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if any(emotion in re.sub(r"[^a-z]", "", name) for emotion in EMOTION_NAMES):
            cols.append(col)
    return cols


def main():
    print("=" * 72)
    print("PHASE 6C — LINGUISTIC FEATURES × SENTIMENT / EMOTION")
    print("=" * 72)

    for path in [FEATURE_FILE, VADER_FILE, EMOTION_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}. Run Phase 6A and the prior "
                "VADER/transformer analyses first."
            )

    linguistic = pd.read_csv(FEATURE_FILE, low_memory=False)
    vader = pd.read_csv(VADER_FILE, low_memory=False)
    emotion = pd.read_csv(EMOTION_FILE, low_memory=False)

    for name, df in [("linguistic", linguistic), ("VADER", vader), ("transformer", emotion)]:
        if "record_id" not in df.columns:
            raise ValueError(f"'record_id' is missing from the {name} results.")
        if df["record_id"].duplicated().any():
            raise ValueError(f"Duplicate record_id values found in {name} results.")

    compound_col = find_column(
        vader, ["vader_compound", "compound", "vader_compound_score"]
    )
    vader_label_col = find_column(vader, ["vader_label", "sentiment_label", "label"])
    emotion_label_col = find_column(
        emotion, ["emotion_label", "predicted_emotion", "label"]
    )
    confidence_col = find_column(
        emotion, ["emotion_confidence", "confidence", "max_probability"]
    )

    if compound_col is None:
        raise ValueError(
            "Could not identify VADER compound score. VADER columns are: "
            + ", ".join(vader.columns.astype(str))
        )
    if emotion_label_col is None:
        raise ValueError(
            "Could not identify transformer emotion label. Transformer columns are: "
            + ", ".join(emotion.columns.astype(str))
        )

    emotion_prob_cols = identify_emotion_probability_columns(emotion)
    # Require the familiar seven emotion dimensions when those probability
    # columns are available; otherwise report the available columns for review.
    print(f"\nLinguistic rows: {len(linguistic):,}")
    print(f"VADER rows: {len(vader):,}")
    print(f"Transformer rows: {len(emotion):,}")
    print(f"Detected VADER compound column: {compound_col}")
    print(f"Detected emotion label column: {emotion_label_col}")
    print(f"Detected emotion probability columns: {emotion_prob_cols or 'none'}")

    if not emotion_prob_cols:
        raise ValueError(
            "No numeric emotion-probability columns were detected. Inspect the "
            "transformer CSV column names and share them so the script can be "
            "adapted without guessing."
        )

    vader_keep = ["record_id", compound_col]
    if vader_label_col:
        vader_keep.append(vader_label_col)
    emotion_keep = ["record_id", emotion_label_col] + emotion_prob_cols
    if confidence_col and confidence_col not in emotion_keep:
        emotion_keep.append(confidence_col)

    # Keep metadata from the linguistic table and attach only relevant model outputs.
    merged = linguistic.merge(
        vader[vader_keep], on="record_id", how="inner", validate="one_to_one"
    )
    merged = merged.merge(
        emotion[emotion_keep], on="record_id", how="inner", validate="one_to_one",
        suffixes=("", "_transformer")
    )

    if len(merged) != len(linguistic):
        raise ValueError(
            f"Merge retained {len(merged):,} rows from {len(linguistic):,} "
            "linguistic rows. Check record_id matching before proceeding."
        )

    print(f"Matched observations: {len(merged):,}")

    missing_features = [f for f in LINGUISTIC_FEATURES if f not in merged.columns]
    if missing_features:
        raise ValueError("Missing linguistic features: " + ", ".join(missing_features))

    targets = [compound_col] + emotion_prob_cols
    if confidence_col and confidence_col in merged.columns:
        targets.append(confidence_col)

    # Spearman associations: robust to skew and rank-based.
    corr_rows = []
    for feature in LINGUISTIC_FEATURES:
        for target in targets:
            pair = merged[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 3 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(pair[feature], pair[target])
            corr_rows.append({
                "feature": feature,
                "target": target,
                "n": len(pair),
                "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                "p_raw": float(p) if pd.notna(p) else np.nan,
            })

    correlations = pd.DataFrame(corr_rows)
    valid = correlations["p_raw"].notna()
    correlations["p_bh_fdr"] = np.nan
    correlations.loc[valid, "p_bh_fdr"] = bh_adjust(
        correlations.loc[valid, "p_raw"].to_numpy()
    )
    correlations["significant_fdr_0_05"] = correlations["p_bh_fdr"] < 0.05
    correlations["absolute_rho"] = correlations["spearman_rho"].abs()
    correlations = correlations.sort_values(
        ["p_bh_fdr", "absolute_rho"], ascending=[True, False], na_position="last"
    )

    corr_path = OUTPUT_DIR / "linguistic_sentiment_emotion_correlations.csv"
    correlations.to_csv(corr_path, index=False)

    # Descriptive summary by predicted emotion label.
    group_rows = []
    for emotion_label, group in merged.groupby(emotion_label_col, dropna=False):
        for feature in LINGUISTIC_FEATURES:
            values = group[feature].dropna()
            group_rows.append({
                "emotion_label": emotion_label,
                "feature": feature,
                "n": len(values),
                "mean": float(values.mean()) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
            })
    emotion_summary = pd.DataFrame(group_rows)
    summary_path = OUTPUT_DIR / "linguistic_features_by_emotion_summary.csv"
    emotion_summary.to_csv(summary_path, index=False)

    # Comparable-window sensitivity analysis for the correlation estimates.
    if "year" in merged.columns:
        year_values = pd.to_numeric(merged["year"], errors="coerce")
        df_2016 = merged.loc[year_values == 2016].copy()
        if len(df_2016) > 2:
            rows_2016 = []
            for feature in LINGUISTIC_FEATURES:
                for target in targets:
                    pair = df_2016[[feature, target]].replace(
                        [np.inf, -np.inf], np.nan
                    ).dropna()
                    if len(pair) < 3 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
                        rho, p = np.nan, np.nan
                    else:
                        rho, p = spearmanr(pair[feature], pair[target])
                    rows_2016.append({
                        "feature": feature,
                        "target": target,
                        "n": len(pair),
                        "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                        "p_raw": float(p) if pd.notna(p) else np.nan,
                    })
            corr_2016 = pd.DataFrame(rows_2016)
            valid_2016 = corr_2016["p_raw"].notna()
            corr_2016["p_bh_fdr"] = np.nan
            corr_2016.loc[valid_2016, "p_bh_fdr"] = bh_adjust(
                corr_2016.loc[valid_2016, "p_raw"].to_numpy()
            )
            corr_2016["significant_fdr_0_05"] = corr_2016["p_bh_fdr"] < 0.05
            corr_2016["absolute_rho"] = corr_2016["spearman_rho"].abs()
            corr_2016 = corr_2016.sort_values(
                ["p_bh_fdr", "absolute_rho"], ascending=[True, False], na_position="last"
            )
            corr_2016.to_csv(
                OUTPUT_DIR / "linguistic_sentiment_emotion_correlations_2016.csv",
                index=False
            )

    print("\nTop 25 associations by adjusted p-value:")
    print(
        correlations[
            ["feature", "target", "n", "spearman_rho", "p_raw",
             "p_bh_fdr", "significant_fdr_0_05"]
        ].head(25).round(6).to_string(index=False)
    )
    print("\nSaved:")
    print(f"  {corr_path}")
    print(f"  {summary_path}")
    if "year" in merged.columns and "df_2016" in locals() and len(df_2016) > 2:
        print(f"  {OUTPUT_DIR / 'linguistic_sentiment_emotion_correlations_2016.csv'}")

    print("\nInterpretation notes:")
    print("- Spearman rho measures monotonic association, not causation.")
    print("- FDR-adjusted p-values address the multiple correlation tests.")
    print("- Small correlations may be statistically significant in this sample.")
    print("- Emotion probabilities are model outputs, not ground-truth labels.")
    print("- No user-level psychological or diagnostic inference is made.")
    print("- Source corpus and earlier result files were not modified.")


if __name__ == "__main__":
    main()
