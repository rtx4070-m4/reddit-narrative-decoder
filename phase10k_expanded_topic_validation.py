
from pathlib import Path
import pandas as pd

# ============================================================
# PHASE 10K: EXPANDED TOPIC VALIDATION SAMPLE
# Up to 20 records per model-community-topic group.
# Deterministic sampling; no retraining or source modifications.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

TEXT_FILE = DATA / "reddit_nlp_corpus.csv"
PHASE8_FILE = TABLES / "phase8_cross_community" / "phase8_merged_observations.csv"

MODEL_DIR = TABLES / "phase7o_temporal_narratives_v17"

MODELS = {
    "NMF": MODEL_DIR / "nmf_document_topic_date_join.csv",
    "Semantic_KMeans": MODEL_DIR / "semantic_kmeans_document_topic_date_join.csv",
}

OUT_DIR = TABLES / "phase10k_expanded_topic_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 20
RANDOM_STATE = 2026


def find_text_column(df):
    for col in ["analysis_text", "text", "body", "selftext"]:
        if col in df.columns:
            return col
    raise ValueError(
        "Text column not found. Available columns: "
        + ", ".join(df.columns)
    )


def main():
    print("=" * 76)
    print("PHASE 10K: EXPANDED TOPIC VALIDATION")
    print("=" * 76)

    for path in [TEXT_FILE, PHASE8_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    phase8 = pd.read_csv(PHASE8_FILE, low_memory=False)
    text_df = pd.read_csv(TEXT_FILE, low_memory=False)

    text_col = find_text_column(text_df)

    for df in [phase8, text_df]:
        df["record_id"] = df["record_id"].astype(str)

    if phase8["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in Phase 8 corpus.")

    if text_df["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in NLP corpus.")

    text_lookup = text_df[["record_id", text_col]].rename(
        columns={text_col: "review_text"}
    )
    text_lookup["review_text"] = (
        text_lookup["review_text"].fillna("").astype(str).str.strip()
    )

    corpus = phase8[
        ["record_id", "community", "created_utc"]
    ].merge(
        text_lookup,
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    corpus["audit_month"] = pd.to_datetime(
        corpus["created_utc"], errors="coerce", utc=True
    ).dt.strftime("%Y-%m")

    # Keep February 2016 to match the previous Phase 10F/10G comparisons.
    feb = corpus[
        (corpus["audit_month"] == "2016-02")
        & corpus["review_text"].ne("")
    ].copy()

    print("Phase 8 records:", len(phase8))
    print("NLP corpus records:", len(text_df))
    print("February records with usable text:", len(feb))
    print("Text column:", text_col)

    all_samples = []
    summary_rows = []

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model assignment file not found: {model_path}")

        assignments = pd.read_csv(model_path, low_memory=False)
        assignments["record_id"] = assignments["record_id"].astype(str)

        if assignments["record_id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {model_name} assignments.")

        if "topic_label" not in assignments.columns:
            raise ValueError(f"topic_label missing from {model_path}")

        merged = feb.merge(
            assignments[["record_id", "topic_label"]],
            on="record_id",
            how="inner",
            validate="one_to_one",
        )

        labels = merged["topic_label"].fillna("").astype(str).str.strip()
        assigned = (
            labels.ne("")
            & ~labels.str.upper().isin(["NAN", "NONE", "MISSING_TOPIC"])
            & ~labels.str.upper().str.contains(
                r"UNASSIGNED|UNASSIGN|NO_TOPIC", regex=True, na=False
            )
        )

        merged = merged.loc[assigned].copy()
        merged["model"] = model_name
        merged["topic"] = merged["topic_label"].astype(str)

        print(f"\n{model_name}: assigned February records = {len(merged)}")

        for (community, topic), group in merged.groupby(
            ["community", "topic"], sort=True
        ):
            n_available = len(group)
            n_sample = min(SAMPLE_SIZE, n_available)

            sample = group.sample(
                n=n_sample, random_state=RANDOM_STATE
            ).copy()

            sample["available_assigned_examples"] = n_available
            sample["sample_size_for_group"] = n_sample
            sample["human_topic_label"] = ""
            sample["interpretability_rating"] = ""
            sample["evidence_notes"] = ""
            sample["contradictory_examples"] = ""

            all_samples.append(sample)

            summary_rows.append({
                "model": model_name,
                "community": community,
                "topic": topic,
                "available_assigned_examples": n_available,
                "examples_sampled": n_sample,
                "sample_fraction": n_sample / n_available,
            })

    if not all_samples:
        raise ValueError("No assigned topic examples available for review.")

    samples = pd.concat(all_samples, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    sample_columns = [
        "model", "community", "topic", "record_id", "review_text",
        "available_assigned_examples", "sample_size_for_group",
        "human_topic_label", "interpretability_rating",
        "evidence_notes", "contradictory_examples",
    ]

    sample_path = OUT_DIR / "phase10k_expanded_sample.csv"
    summary_path = OUT_DIR / "phase10k_sampling_summary.csv"

    samples[sample_columns].to_csv(sample_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSampling summary:")
    print(summary.to_string(index=False))

    print("\nTotal sampled records:", len(samples))
    print("Number of groups:", len(summary))
    print("Saved:", sample_path)
    print("Saved:", summary_path)

    print("\nReview guidance:")
    print("- Use descriptive labels grounded in repeated subjects.")
    print("- Rate each group clear, mixed, or unclear.")
    print("- Record examples that contradict the proposed label.")
    print("- Do not infer psychological traits about individual authors.")
    print("- Samples are for qualitative validation, not ground-truth labels.")

    print("\n" + "=" * 76)
    print("PHASE 10K COMPLETED")
    print("Existing model assignments and source files were not modified.")
    print("=" * 76)


if __name__ == "__main__":
    main()
