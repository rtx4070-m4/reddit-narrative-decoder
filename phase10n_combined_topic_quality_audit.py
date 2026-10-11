
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"

TFIDF_FILE = (
    TABLES / "phase10l_coherence_diagnostics"
    / "phase10l_topic_coherence_diagnostics.csv"
)
SEMANTIC_FILE = (
    TABLES / "phase10m_semantic_coherence"
    / "phase10m_topic_semantic_similarity.csv"
)
SAMPLING_FILE = (
    TABLES / "phase10k_expanded_topic_validation"
    / "phase10k_sampling_summary.csv"
)

OUT_DIR = TABLES / "phase10n_combined_topic_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["model", "community", "topic"]


def main():
    print("=" * 76)
    print("PHASE 10N: COMBINED TOPIC-QUALITY AUDIT")
    print("=" * 76)

    for path in [TFIDF_FILE, SEMANTIC_FILE, SAMPLING_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    tfidf = pd.read_csv(TFIDF_FILE, low_memory=False)
    semantic = pd.read_csv(SEMANTIC_FILE, low_memory=False)
    sampling = pd.read_csv(SAMPLING_FILE, low_memory=False)

    for name, df in [
        ("TF-IDF", tfidf),
        ("Semantic", semantic),
        ("Sampling", sampling),
    ]:
        missing = set(KEYS) - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} file missing columns: {sorted(missing)}"
            )
        if df.duplicated(KEYS).any():
            raise ValueError(
                f"{name} file has duplicate model-community-topic keys."
            )

    # Retain only the relevant TF-IDF fields.
    tfidf_columns = KEYS + [
        col for col in [
            "sample_size",
            "mean_pairwise_tfidf_similarity",
            "median_pairwise_tfidf_similarity",
            "distinctive_terms",
            "diagnostic_note",
        ]
        if col in tfidf.columns
    ]
    tfidf = tfidf[tfidf_columns].copy()

    semantic_columns = KEYS + [
        col for col in [
            "sample_size",
            "available_assigned_examples",
            "mean_pairwise_semantic_similarity",
            "median_pairwise_semantic_similarity",
            "mean_similarity_to_group_centroid",
            "diagnostic_note",
        ]
        if col in semantic.columns
    ]
    semantic = semantic[semantic_columns].copy()

    # Rename overlapping columns before joining.
    tfidf = tfidf.rename(columns={
        col: f"tfidf_{col}"
        for col in tfidf.columns if col not in KEYS
    })
    semantic = semantic.rename(columns={
        col: f"semantic_{col}"
        for col in semantic.columns if col not in KEYS
    })

    combined = tfidf.merge(
        semantic, on=KEYS, how="outer", validate="one_to_one"
    )

    combined = combined.merge(
        sampling, on=KEYS, how="outer", validate="one_to_one",
        suffixes=("", "_sampling")
    )

    if len(combined) != 40:
        raise ValueError(
            f"Expected 40 topic groups, found {len(combined)}."
        )

    # Rank each metric within its own model. Higher means relatively
    # higher among that model's 20 groups, not necessarily "good."
    metrics = [
        ("tfidf_mean_pairwise_tfidf_similarity", "tfidf_rank_percentile"),
        (
            "semantic_mean_pairwise_semantic_similarity",
            "semantic_rank_percentile",
        ),
        (
            "semantic_mean_similarity_to_group_centroid",
            "centroid_rank_percentile",
        ),
    ]

    for metric, rank_col in metrics:
        if metric in combined.columns:
            combined[rank_col] = combined.groupby("model")[metric].rank(
                pct=True, method="average"
            )

    sample_col = (
        "examples_sampled"
        if "examples_sampled" in combined.columns
        else None
    )

    if sample_col:
        combined["sample_size_warning"] = combined[sample_col].apply(
            lambda n: (
                "Very small sample" if pd.notna(n) and n < 5
                else "Small sample" if pd.notna(n) and n < 10
                else "Review normally"
            )
        )

    output_path = OUT_DIR / "phase10n_combined_topic_quality.csv"
    combined.to_csv(output_path, index=False)

    summary_rows = []
    for model_name, group in combined.groupby("model", sort=True):
        summary_rows.append({
            "model": model_name,
            "topic_groups": len(group),
            "mean_tfidf_similarity": group[
                "tfidf_mean_pairwise_tfidf_similarity"
            ].mean(),
            "median_tfidf_similarity": group[
                "tfidf_mean_pairwise_tfidf_similarity"
            ].median(),
            "mean_semantic_similarity": group[
                "semantic_mean_pairwise_semantic_similarity"
            ].mean(),
            "median_semantic_group_similarity": group[
                "semantic_mean_pairwise_semantic_similarity"
            ].median(),
            "mean_centroid_similarity": group[
                "semantic_mean_similarity_to_group_centroid"
            ].mean(),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "phase10n_model_quality_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Topic groups:", len(combined))
    print("\nModel summary:")
    print(summary.to_string(index=False))

    print("\nHighest semantic-similarity groups per model:")
    for model_name, group in combined.groupby("model", sort=True):
        metric = "semantic_mean_pairwise_semantic_similarity"
        print(f"\n{model_name}")
        print(
            group.sort_values(metric, ascending=False)[
                ["community", "topic", metric]
            ].head(5).to_string(index=False)
        )

    print("\nSaved:", output_path)
    print("Saved:", summary_path)
    print("\nInterpretation:")
    print("- Percentile ranks compare groups within each model only.")
    print("- Similarity scores are not validated topic-quality scores.")
    print("- Final selection must also use human review and topic meaning.")

    print("\n" + "=" * 76)
    print("PHASE 10N COMPLETED")
    print("=" * 76)


if __name__ == "__main__":
    main()
