
from pathlib import Path
import pandas as pd

# ============================================================
# PHASE 10B: DUPLICATE-TEXT AND TOPIC-QUALITY AUDIT
# Project: Reddit Narrative Decoder
# Read-only audit: existing data and models are not modified.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

NLP_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "reddit_nlp_corpus.csv"
)

PHASE8_PATH = (
    PROJECT_ROOT / "results" / "tables"
    / "phase8_cross_community"
    / "phase8_merged_observations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "tables"
    / "phase10b_topic_quality_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 75)
    print("PHASE 10B: DUPLICATE-TEXT AUDIT")
    print("=" * 75)
    print("Project root:", PROJECT_ROOT)

    if not NLP_PATH.exists():
        raise FileNotFoundError(
            f"NLP corpus not found: {NLP_PATH}\n"
            "Check that you are using the correct project folder."
        )

    # --------------------------------------------------------
    # 1. Load the existing NLP corpus
    # --------------------------------------------------------
    corpus = pd.read_csv(NLP_PATH, low_memory=False)

    print("\n[1] CORPUS INFORMATION")
    print("Input file:", NLP_PATH)
    print("Rows:", len(corpus))
    print("Columns:", corpus.columns.tolist())

    # Prefer the original text column where available.
    candidates = [
        "text", "original_text",
        "analysis_text", "topic7k_clean_text"
    ]
    text_col = next(
        (column for column in candidates if column in corpus.columns),
        None
    )

    if text_col is None:
        raise ValueError(
            "No usable text column found. Available columns: "
            + ", ".join(corpus.columns)
        )

    if "community" not in corpus.columns:
        if "subreddit" in corpus.columns:
            corpus["community"] = corpus["subreddit"]
        else:
            raise ValueError(
                "Neither 'community' nor 'subreddit' exists."
            )

    if "record_id" not in corpus.columns:
        raise ValueError("The required 'record_id' column is missing.")

    print("Text column used:", text_col)

    # Normalize case and whitespace for comparison only.
    # Original source text remains unchanged.
    corpus["normalized_text_for_audit"] = (
        corpus[text_col]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    audit = corpus[
        corpus["normalized_text_for_audit"].ne("")
    ].copy()

    print("Non-empty text rows:", len(audit))
    print("Empty text rows excluded:", len(corpus) - len(audit))

    # --------------------------------------------------------
    # 2. Duplicate-text summary within each community
    # --------------------------------------------------------
    summary = (
        audit.groupby("community", dropna=False)
        .agg(
            records=("record_id", "size"),
            unique_normalized_texts=(
                "normalized_text_for_audit", "nunique"
            ),
            unique_record_ids=("record_id", "nunique"),
        )
        .reset_index()
    )

    summary["records_beyond_unique_texts"] = (
        summary["records"] - summary["unique_normalized_texts"]
    )

    summary["duplicate_text_excess_share"] = (
        summary["records_beyond_unique_texts"]
        / summary["records"].replace(0, pd.NA)
    )

    summary = summary.sort_values(
        "records_beyond_unique_texts", ascending=False
    )

    summary_path = OUTPUT_DIR / "duplicate_text_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n[2] DUPLICATE-TEXT SUMMARY BY COMMUNITY")
    print(summary.to_string(index=False))
    print("Saved:", summary_path)

    # --------------------------------------------------------
    # 3. Repeated text within each community
    # --------------------------------------------------------
    groups = (
        audit.groupby(
            ["community", "normalized_text_for_audit"],
            dropna=False
        )
        .agg(
            record_count=("record_id", "size"),
            record_ids=(
                "record_id",
                lambda values: " | ".join(
                    values.astype(str).head(10)
                )
            ),
            example_original_text=(text_col, "first"),
        )
        .reset_index()
    )

    groups = groups[groups["record_count"] > 1].copy()
    groups = groups.sort_values(
        ["record_count", "community"],
        ascending=[False, True]
    )

    groups_path = OUTPUT_DIR / "duplicate_text_groups.csv"
    groups.to_csv(groups_path, index=False)

    print("\n[3] LARGEST REPEATED-TEXT GROUPS WITHIN COMMUNITIES")
    if groups.empty:
        print("No repeated normalized text found within communities.")
    else:
        print(
            groups[
                [
                    "community", "record_count",
                    "record_ids", "example_original_text"
                ]
            ].head(15).to_string(index=False, max_colwidth=180)
        )

    print("Total repeated-text groups:", len(groups))
    print("Saved:", groups_path)

    # --------------------------------------------------------
    # 4. Identical normalized text across communities
    # --------------------------------------------------------
    cross = (
        audit.groupby("normalized_text_for_audit")
        .agg(
            record_count=("record_id", "size"),
            community_count=("community", "nunique"),
            communities=(
                "community",
                lambda values: " | ".join(
                    sorted(set(values.astype(str)))
                )
            ),
            record_ids=(
                "record_id",
                lambda values: " | ".join(
                    values.astype(str).head(10)
                )
            ),
            example_original_text=(text_col, "first"),
        )
        .reset_index()
    )

    cross = cross[
        (cross["record_count"] > 1)
        & (cross["community_count"] > 1)
    ].sort_values(
        ["community_count", "record_count"],
        ascending=False
    )

    cross_path = (
        OUTPUT_DIR / "cross_community_duplicate_text_groups.csv"
    )
    cross.to_csv(cross_path, index=False)

    print("\n[4] REPEATED TEXT ACROSS COMMUNITIES")
    if cross.empty:
        print("No cross-community repeated text found.")
    else:
        print(
            cross[
                [
                    "record_count", "community_count",
                    "communities", "example_original_text"
                ]
            ].head(15).to_string(index=False, max_colwidth=180)
        )

    print("Cross-community repeated-text groups:", len(cross))
    print("Saved:", cross_path)

    # --------------------------------------------------------
    # 5. Optional reconciliation with Phase 8
    # --------------------------------------------------------
    print("\n[5] PHASE 8 RECORD-ID RECONCILIATION")

    if PHASE8_PATH.exists():
        phase8 = pd.read_csv(PHASE8_PATH, low_memory=False)

        print("Phase 8 rows:", len(phase8))
        print("Phase 8 columns:", phase8.columns.tolist())

        if "record_id" in phase8.columns:
            phase8_ids = phase8[["record_id"]].drop_duplicates()

            text_lookup = (
                corpus[["record_id", "community", text_col]]
                .drop_duplicates("record_id")
            )

            joined = phase8_ids.merge(
                text_lookup,
                on="record_id",
                how="left",
                indicator=True
            )

            print("\nRecord-ID matching status:")
            print(joined["_merge"].value_counts().to_string())

            reconciliation_path = (
                OUTPUT_DIR / "phase8_text_id_reconciliation.csv"
            )
            joined.to_csv(reconciliation_path, index=False)
            print("Saved:", reconciliation_path)
        else:
            print("Phase 8 file has no record_id column; skipped join.")
    else:
        print("Phase 8 file not found; optional reconciliation skipped.")

    # --------------------------------------------------------
    # 6. Finish
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("PHASE 10B AUDIT COMPLETED")
    print("=" * 75)
    print("Output directory:", OUTPUT_DIR)
    print("\nFiles created:")
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f" - {path.name}")

    print(
        "\nIMPORTANT: Repeated text does not automatically mean "
        "duplicate records. Do not delete records based on text "
        "matching alone."
    )


if __name__ == "__main__":
    main()
