
from pathlib import Path
import pandas as pd

# ============================================================
# PHASE 10G: TOPIC INTERPRETABILITY AUDIT
# Corrected: load original text from the NLP corpus.
# Join keys: record_id. No model retraining or source changes.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

CORPUS_FILE = TABLES / "phase8_cross_community" / "phase8_merged_observations.csv"
TEXT_FILE = DATA / "reddit_nlp_corpus.csv"

MODEL_DIR = TABLES / "phase7o_temporal_narratives_v17"
OUT_DIR = TABLES / "phase10g_topic_interpretability_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "NMF": MODEL_DIR / "nmf_document_topic_date_join.csv",
    "Semantic_KMeans": MODEL_DIR / "semantic_kmeans_document_topic_date_join.csv",
}

SAMPLES_PER_CELL = 5
RANDOM_STATE = 42


def find_text_column(df):
    for col in ["analysis_text", "text", "body", "selftext"]:
        if col in df.columns:
            return col
    raise ValueError(
        "No text column found. Available columns: "
        + ", ".join(df.columns)
    )


def main():
    print("=" * 75)
    print("PHASE 10G: TOPIC INTERPRETABILITY AUDIT")
    print("=" * 75)

    for path in [CORPUS_FILE, TEXT_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    phase8 = pd.read_csv(CORPUS_FILE, low_memory=False)
    text_df = pd.read_csv(TEXT_FILE, low_memory=False)

    required_phase8 = {"record_id", "community", "created_utc"}
    missing = required_phase8 - set(phase8.columns)
    if missing:
        raise ValueError(f"Phase 8 file is missing columns: {sorted(missing)}")

    if "record_id" not in text_df.columns:
        raise ValueError("NLP corpus does not contain record_id.")

    text_col = find_text_column(text_df)

    phase8["record_id"] = phase8["record_id"].astype(str)
    text_df["record_id"] = text_df["record_id"].astype(str)

    if phase8["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id values in Phase 8 corpus.")

    if text_df["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id values in NLP corpus.")

    # Use original text only for records present in Phase 8.
    text_lookup = text_df[["record_id", text_col]].copy()
    text_lookup = text_lookup.rename(columns={text_col: "review_text"})
    text_lookup["review_text"] = (
        text_lookup["review_text"].fillna("").astype(str).str.strip()
    )

    corpus = phase8.merge(
        text_lookup,
        on="record_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched_text = int((corpus["_merge"] == "both").sum())
    missing_text = int((corpus["_merge"] == "left_only").sum())

    print("Phase 8 records:", len(phase8))
    print("NLP corpus records:", len(text_df))
    print("Records matched to text:", matched_text)
    print("Records without matched text:", missing_text)
    print("Text column selected:", text_col)

    corpus = corpus.drop(columns="_merge")
    corpus["audit_month"] = pd.to_datetime(
        corpus["created_utc"], errors="coerce", utc=True
    ).dt.strftime("%Y-%m")

    feb = corpus[corpus["audit_month"] == "2016-02"].copy()
    feb = feb[feb["review_text"].ne("")].copy()

    print("February records with usable text:", len(feb))

    all_samples = []
    summary_rows = []

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Topic assignment file not found: {model_path}")

        model = pd.read_csv(model_path, low_memory=False)

        if "record_id" not in model.columns or "topic_label" not in model.columns:
            raise ValueError(
                f"{model_name} assignment file must contain record_id and topic_label."
            )

        model["record_id"] = model["record_id"].astype(str)

        if model["record_id"].duplicated().any():
            raise ValueError(f"Duplicate record IDs in {model_name} assignments.")

        merged = feb[
            ["record_id", "community", "audit_month", "review_text"]
        ].merge(
            model[["record_id", "topic_label"]],
            on="record_id",
            how="inner",
            validate="one_to_one",
        )

        labels = merged["topic_label"].fillna("").astype(str).str.strip()
        valid = (
            labels.ne("")
            & ~labels.str.upper().isin(["NAN", "NONE", "MISSING_TOPIC"])
            & ~labels.str.upper().str.contains(
                r"UNASSIGNED|UNASSIGN|NO_TOPIC", regex=True, na=False
            )
        )

        merged = merged.loc[valid & merged["review_text"].ne("")].copy()
        merged["model"] = model_name
        merged["topic"] = merged["topic_label"].astype(str)

        print(f"\n{model_name}: eligible assigned February records = {len(merged)}")

        for (community, topic), group in merged.groupby(
            ["community", "topic"], sort=True
        ):
            n_available = len(group)
            n_sample = min(SAMPLES_PER_CELL, n_available)

            sample = group.sample(
                n=n_sample, random_state=RANDOM_STATE
            ).copy()

            sample["available_assigned_examples"] = n_available
            sample["sample_size_for_cell"] = n_sample
            sample["human_topic_label"] = ""
            sample["interpretability_rating"] = ""
            sample["review_notes"] = ""

            all_samples.append(sample)

            summary_rows.append({
                "model": model_name,
                "community": community,
                "topic": topic,
                "available_assigned_examples": n_available,
                "examples_sampled": n_sample,
            })

    if not all_samples:
        raise ValueError(
            "No eligible examples found. Check record_id matching and topic labels."
        )

    samples = pd.concat(all_samples, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    sample_columns = [
        "model", "community", "topic", "record_id", "review_text",
        "available_assigned_examples", "sample_size_for_cell",
        "human_topic_label", "interpretability_rating", "review_notes",
    ]

    sample_path = OUT_DIR / "phase10g_sampled_topic_examples.csv"
    summary_path = OUT_DIR / "phase10g_sampling_summary.csv"

    samples[sample_columns].to_csv(sample_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSampling summary:")
    print(summary.to_string(index=False))

    print("\nSaved:", sample_path)
    print("Sample rows:", len(samples))
    print("Saved:", summary_path)
    print("Summary rows:", len(summary))

    print("\nHuman review instructions:")
    print("1. Read examples grouped by model, community, and topic.")
    print("2. Enter a short descriptive label in human_topic_label.")
    print("3. Use clear, mixed, or unclear in interpretability_rating.")
    print("4. Note generic wording or preprocessing artifacts.")
    print("5. Do not infer psychological traits about individual authors.")

    print("\n" + "=" * 75)
    print("PHASE 10G COMPLETED")
    print("Output directory:", OUT_DIR)
    print("Source files were not modified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
