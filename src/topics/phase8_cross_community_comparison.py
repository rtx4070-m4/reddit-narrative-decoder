"""
Phase 8: Cross-community narrative comparison
Project: Computational Analysis of Emotional Expression, Psychological Language,
and Narrative Evolution in Reddit Communities.

Run from the project root:
    python src/phase8_cross_community_comparison.py

Uses existing model outputs; does not rerun NLP models.
Primary inferential comparison: February 2016 only.
All-years summaries are included as descriptive/sensitivity outputs.

Ethical framing:
- Aggregated community-level language patterns only.
- Lexical markers and model predictions are not validated psychological traits,
  diagnoses, or ground truth.
- Associations are descriptive, not causal.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2_contingency, mannwhitneyu

ROOT = Path(".")
TABLES = ROOT / "results" / "tables"
OUT = TABLES / "phase8_cross_community"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "vader": TABLES / "vader_sentiment_results.csv",
    "emotion": TABLES / "transformer_emotion_results.csv",
    "linguistic": TABLES / "linguistic_features.csv",
    "nmf_topic": TABLES / "phase7o_temporal_narratives_v17" / "nmf_document_topic_date_join.csv",
    "semantic_topic": TABLES / "phase7o_temporal_narratives_v17" / "semantic_kmeans_document_topic_date_join.csv",
}

COMMUNITIES = ["gaming", "science", "todayilearned", "worldnews"]
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LINGUISTIC_FEATURES = [
    "word_count", "lexical_diversity",
    "first_person_ratio", "second_person_ratio", "third_person_ratio",
    "negation_ratio", "cognitive_ratio", "uncertainty_ratio", "social_ratio",
    "achievement_ratio", "risk_threat_ratio", "intensifier_ratio",
    "hedge_ratio", "question_word_ratio",
]
VADER_FEATURES = [
    "vader_compound", "vader_negative", "vader_neutral", "vader_positive",
]
EMOTION_PROBS = [f"emotion_prob_{e}" for e in EMOTIONS]


def load_csv(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run this script from the project root and verify the existing outputs."
            )
        print(f"[INFO] Optional file not found; skipping: {path}")
        return None
    return pd.read_csv(path, low_memory=False)


def normalize_keys(df, name):
    if "record_id" not in df.columns:
        raise ValueError(f"{name} is missing record_id")
    if df["record_id"].duplicated().any():
        n = int(df["record_id"].duplicated().sum())
        raise ValueError(f"{name} has {n} duplicate record_id values; refusing unsafe merge.")
    return df


def date_mask(df, date_col="created_utc"):
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        return dt.dt.year.eq(2016)
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce").eq(2016)
    return pd.Series(True, index=df.index)


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if not len(pv):
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * ranked)
    adjusted = np.minimum(adjusted, 1.0)
    inv = np.empty(m, dtype=int)
    inv[order] = np.arange(m)
    out[valid] = adjusted[inv]
    return out


def epsilon_squared(h, n, k):
    if n <= k:
        return np.nan
    return max(0.0, (h - k + 1) / (n - k))


def kw_tests(df, features, period):
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        groups = []
        used_n = 0
        for community in COMMUNITIES:
            vals = pd.to_numeric(
                df.loc[df["community"].eq(community), feature], errors="coerce"
            ).dropna().to_numpy()
            if len(vals):
                groups.append(vals)
                used_n += len(vals)
        if len(groups) < 2:
            continue
        try:
            h, p = kruskal(*groups, nan_policy="omit")
            rows.append({
                "period": period, "feature": feature, "n": used_n,
                "groups": len(groups), "kruskal_h": float(h),
                "df": len(groups) - 1, "p_raw": float(p),
                "epsilon_squared": epsilon_squared(float(h), used_n, len(groups)),
            })
        except ValueError as exc:
            warnings.warn(f"Kruskal skipped for {feature}: {exc}")
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm_across_features"] = holm_adjust(result["p_raw"])
        result["significant_holm_0_05"] = result["p_holm_across_features"] < 0.05
    return result


def pairwise_tests(df, features, period):
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        for i, a in enumerate(COMMUNITIES):
            for b in COMMUNITIES[i + 1:]:
                x = pd.to_numeric(df.loc[df.community.eq(a), feature], errors="coerce").dropna()
                y = pd.to_numeric(df.loc[df.community.eq(b), feature], errors="coerce").dropna()
                if not len(x) or not len(y):
                    continue
                try:
                    stat, p = mannwhitneyu(x, y, alternative="two-sided")
                    # Rank-biserial effect from U: positive means values in a tend to be higher.
                    rb = 2 * stat / (len(x) * len(y)) - 1
                    rows.append({
                        "period": period, "feature": feature,
                        "community_a": a, "community_b": b,
                        "n_a": len(x), "n_b": len(y),
                        "median_a": x.median(), "median_b": y.median(),
                        "mean_a": x.mean(), "mean_b": y.mean(),
                        "mannwhitney_u": float(stat), "p_raw": float(p),
                        "rank_biserial_a_vs_b": float(rb),
                    })
                except ValueError as exc:
                    warnings.warn(f"Pairwise test skipped for {feature}, {a}/{b}: {exc}")
    result = pd.DataFrame(rows)
    if not result.empty:
        # Holm correction within each feature, across the six community pairs.
        result["p_holm_within_feature"] = np.nan
        for feature, idx in result.groupby("feature").groups.items():
            result.loc[idx, "p_holm_within_feature"] = holm_adjust(
                result.loc[idx, "p_raw"].to_numpy()
            )
        result["significant_holm_0_05"] = result["p_holm_within_feature"] < 0.05
    return result


def categorical_emotion_test(df, period):
    if "emotion_label" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    ct = pd.crosstab(df["community"], df["emotion_label"])
    ct = ct.reindex(index=[c for c in COMMUNITIES if c in ct.index], fill_value=0)
    chi2, p, dof, expected = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    denom = min(ct.shape[0] - 1, ct.shape[1] - 1)
    v = np.sqrt(chi2 / (n * denom)) if denom > 0 and n > 0 else np.nan
    test = pd.DataFrame([{
        "period": period, "n": n, "chi2": chi2, "df": dof,
        "p_value": p, "cramers_v": v,
    }])
    pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
    pct.index.name = "community"
    return test, pct.reset_index()


def save(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"[SAVED] {path} ({len(df):,} rows)")


def main():
    vader = normalize_keys(load_csv(FILES["vader"]), "VADER")
    emotion = normalize_keys(load_csv(FILES["emotion"]), "Transformer emotion")
    linguistic = normalize_keys(load_csv(FILES["linguistic"]), "Linguistic features")

    # Merge only analytical columns to avoid duplicate text columns and accidental key collisions.
    base_cols = ["record_id", "community"]
    if "created_utc" in linguistic.columns:
        base_cols += ["created_utc"]
    elif "year" in linguistic.columns:
        base_cols += ["year"]

    base = linguistic[[c for c in base_cols if c in linguistic.columns]].copy()
    base = base[base["community"].isin(COMMUNITIES)].copy()

    ling_cols = ["record_id"] + [c for c in LINGUISTIC_FEATURES if c in linguistic.columns]
    base = base.merge(linguistic[ling_cols], on="record_id", how="left", validate="one_to_one")

    vader_cols = ["record_id"] + [c for c in VADER_FEATURES + ["vader_label"] if c in vader.columns]
    base = base.merge(vader[vader_cols], on="record_id", how="left", validate="one_to_one")

    emotion_cols = ["record_id"] + [
        c for c in ["emotion_label", "emotion_confidence"] + EMOTION_PROBS
        if c in emotion.columns
    ]
    base = base.merge(emotion[emotion_cols], on="record_id", how="left", validate="one_to_one")

    if "created_utc" in base.columns:
        base["created_utc"] = pd.to_datetime(base["created_utc"], errors="coerce", utc=True)
        base["year"] = base["created_utc"].dt.year
        base["month"] = base["created_utc"].dt.to_period("M").astype(str)
    elif "year" not in base.columns:
        base["year"] = np.nan

    base["is_2016"] = base["year"].eq(2016)
    base["is_february_2016"] = (
        base["created_utc"].dt.strftime("%Y-%m").eq("2016-02")
        if "created_utc" in base.columns else False
    )

    # Validate joins and preserve an auditable merged dataset.
    join_audit = pd.DataFrame([
        {"source": "linguistic", "rows": len(linguistic), "unique_record_id": linguistic.record_id.nunique()},
        {"source": "vader", "rows": len(vader), "unique_record_id": vader.record_id.nunique()},
        {"source": "emotion", "rows": len(emotion), "unique_record_id": emotion.record_id.nunique()},
        {"source": "merged_common_base", "rows": len(base), "unique_record_id": base.record_id.nunique()},
        {"source": "February_2016", "rows": int(base.is_february_2016.sum()),
         "unique_record_id": int(base.loc[base.is_february_2016, "record_id"].nunique())},
    ])
    save(join_audit, "phase8_join_audit.csv")
    save(base, "phase8_merged_observations.csv")

    # Descriptive summaries: overall and February 2016.
    numeric = [c for c in LINGUISTIC_FEATURES + VADER_FEATURES + EMOTION_PROBS
               if c in base.columns]
    for label, subset in [
        ("all_years", base),
        ("2016_only", base[base["is_2016"]]),
        ("2016_02", base[base["is_february_2016"]]),
    ]:
        desc = subset.groupby("community")[numeric].agg(["count", "mean", "median", "std"])
        desc.columns = [f"{a}_{b}" for a, b in desc.columns]
        desc = desc.reset_index()
        save(desc, f"phase8_descriptives_{label}.csv")

        # Kruskal-Wallis across communities for continuous scores/features.
        kw = kw_tests(subset, numeric, label)
        save(kw, f"phase8_kruskal_{label}.csv")

        # Pairwise Mann-Whitney with Holm correction within each feature.
        pw = pairwise_tests(subset, numeric, label)
        save(pw, f"phase8_pairwise_{label}.csv")

        # Emotion-label composition.
        chi, pct = categorical_emotion_test(subset, label)
        save(chi, f"phase8_emotion_label_chisquare_{label}.csv")
        save(pct, f"phase8_emotion_label_percentages_{label}.csv")

        if "vader_label" in subset.columns:
            sentiment_ct = pd.crosstab(subset["community"], subset["vader_label"])
            sentiment_pct = sentiment_ct.div(
                sentiment_ct.sum(axis=1).replace(0, np.nan), axis=0
            ) * 100
            sentiment_pct.index.name = "community"
            save(sentiment_pct.reset_index(), f"phase8_vader_label_percentages_{label}.csv")

    # Topic prevalence from the explicit Phase 7O.17 selected configuration.
    # These files contain the assignment-to-date joins already selected by the prior script.
    for key, outprefix in [("nmf_topic", "nmf"), ("semantic_topic", "semantic")]:
        topic = load_csv(FILES[key], required=False)
        if topic is None:
            continue
        if not {"community", "topic_label"}.issubset(topic.columns):
            warnings.warn(f"{FILES[key]} lacks community/topic_label; skipped.")
            continue
        # Filter to February 2016 using the joined month field, not file order.
        if "month" in topic.columns:
            feb = topic[topic["month"].astype(str).eq("2016-02")].copy()
        elif "analysis_date" in topic.columns:
            dates = pd.to_datetime(topic["analysis_date"], errors="coerce")
            feb = topic[dates.dt.strftime("%Y-%m").eq("2016-02")].copy()
        else:
            warnings.warn(f"No month/date column in {FILES[key]}; topic summary skipped.")
            continue

        counts = feb.groupby(["community", "topic_label"], dropna=False).size().rename("document_count").reset_index()
        totals = feb.groupby("community").size().rename("assigned_total").reset_index()
        counts = counts.merge(totals, on="community", how="left", validate="many_to_one")
        counts["share_among_assigned"] = counts["document_count"] / counts["assigned_total"]
        counts["model"] = outprefix
        save(counts, f"phase8_{outprefix}_feb2016_topic_prevalence.csv")

    print("\nPhase 8 completed.")
    print(f"Outputs: {OUT.resolve()}")
    print("Primary interpretation period: February 2016.")
    print("Do not interpret statistical significance as practical importance or causality.")


if __name__ == "__main__":
    main()
