from pathlib import Path
import sys
import re
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reddit_master_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reddit_nlp_corpus.csv"
)

TABLES_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MASTER CORPUS
# ============================================================

print("=" * 70)
print("BUILDING NLP-READY CORPUS")
print("=" * 70)

print(f"\nInput: {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["created_utc"]
)

print(f"Initial observations: {len(df):,}")


# ============================================================
# PRESERVE ORIGINAL TEXT
# ============================================================

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
)

# IMPORTANT:
# `text` is never overwritten.
# It remains the original analytical text.


# ============================================================
# IDENTIFY PLACEHOLDERS
# ============================================================

PLACEHOLDER_VALUES = {
    "removed",
    "deleted",
    "[removed]",
    "[deleted]",
    "removed by reddit",
    "deleted by user"
}

normalized_text = (
    df["text"]
    .str.strip()
    .str.lower()
)

df["is_placeholder"] = (
    normalized_text.isin(
        PLACEHOLDER_VALUES
    )
)


# ============================================================
# REMOVE ONLY PLACEHOLDER OBSERVATIONS
# ============================================================

placeholder_count = (
    df["is_placeholder"].sum()
)

print(
    f"Placeholder observations identified: "
    f"{placeholder_count:,}"
)

nlp_df = df[
    ~df["is_placeholder"]
].copy()

print(
    f"Observations after placeholder removal: "
    f"{len(nlp_df):,}"
)


# ============================================================
# CONTROLLED TEXT NORMALIZATION
# ============================================================

# We create a NEW column.
# The original `text` column is untouched.

def normalize_text(text):

    text = str(text)

    # Normalize repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove leading/trailing whitespace.
    text = text.strip()

    return text


nlp_df["analysis_text"] = (
    nlp_df["text"]
    .apply(normalize_text)
)


# ============================================================
# REMOVE EMPTY ANALYSIS TEXT
# ============================================================

empty_analysis = (
    nlp_df["analysis_text"]
    .str.len()
    == 0
)

empty_count = empty_analysis.sum()

print(
    f"Empty analysis texts removed: "
    f"{empty_count:,}"
)

nlp_df = nlp_df[
    ~empty_analysis
].copy()


# ============================================================
# TEXT LENGTH FEATURES
# ============================================================

nlp_df["analysis_char_count"] = (
    nlp_df["analysis_text"]
    .str.len()
)

nlp_df["analysis_word_count"] = (
    nlp_df["analysis_text"]
    .str.split()
    .str.len()
)


# ============================================================
# QUALITY FLAGS
# ============================================================

nlp_df["is_very_short"] = (
    nlp_df["analysis_word_count"] <= 3
)

nlp_df["is_long_text"] = (
    nlp_df["analysis_word_count"] > 250
)

nlp_df["is_very_long_text"] = (
    nlp_df["analysis_word_count"] > 500
)


# ============================================================
# ARTICLE-LIKE FLAG
# ============================================================

ARTICLE_PATTERNS = [
    r"\bfull article\b",
    r"\bpage \d+ of \d+\b",
    r"\bpart \d+ of \d+\b",
    r"\bdownloaded from\b",
    r"\babstract\b",
    r"\bjournal\b",
    r"\bresearch paper\b",
    r"\bdoi\b",
]

article_regex = re.compile(
    "|".join(ARTICLE_PATTERNS),
    flags=re.IGNORECASE
)

nlp_df["is_article_like"] = (
    nlp_df["analysis_text"]
    .apply(
        lambda x: bool(
            article_regex.search(x)
        )
    )
)


# ============================================================
# REPEATED TEXT FLAG
# ============================================================

text_frequency = (
    nlp_df
    .groupby("analysis_text")["record_id"]
    .transform("count")
)

nlp_df["text_frequency"] = (
    text_frequency
)

nlp_df["is_repeated_text"] = (
    nlp_df["text_frequency"] > 1
)


# ============================================================
# SENTENCE BOUNDARY
# ============================================================

nlp_df["sentence_boundary_count"] = (
    nlp_df["analysis_text"]
    .apply(
        lambda x: len(
            re.findall(
                r"[.!?]+",
                x
            )
        )
    )
)

nlp_df["has_sentence_boundary"] = (
    nlp_df["sentence_boundary_count"] > 0
)


# ============================================================
# ANALYSIS VERSION
# ============================================================

nlp_df["corpus_version"] = (
    "nlp_v1"
)


# ============================================================
# ANALYSIS PURPOSE
# ============================================================

nlp_df["analysis_eligible"] = True


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================

final_columns = [
    "record_id",
    "id",
    "community",
    "subreddit",
    "created_utc",
    "date",
    "year",
    "month",
    "day",
    "hour",
    "text",
    "analysis_text",
    "meta",
    "ups",
    "downs",
    "source_file",

    # Quality / analysis metadata
    "is_very_short",
    "is_long_text",
    "is_very_long_text",
    "is_article_like",
    "text_frequency",
    "is_repeated_text",
    "sentence_boundary_count",
    "has_sentence_boundary",
    "corpus_version",
    "analysis_eligible",
]


nlp_df = nlp_df[
    final_columns
]


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("NLP CORPUS VALIDATION")
print("=" * 70)

print(
    f"\nFinal observations: "
    f"{len(nlp_df):,}"
)

print(
    f"Expected observations: "
    f"{len(df) - placeholder_count - empty_count:,}"
)

print(
    f"Unique record IDs: "
    f"{nlp_df['record_id'].nunique():,}"
)

print(
    f"Unique source IDs: "
    f"{nlp_df['id'].nunique():,}"
)

print(
    f"Unique communities: "
    f"{nlp_df['community'].nunique():,}"
)

print(
    f"Missing original text: "
    f"{nlp_df['text'].isna().sum():,}"
)

print(
    f"Missing analysis text: "
    f"{nlp_df['analysis_text'].isna().sum():,}"
)


# ============================================================
# COMMUNITY COUNTS
# ============================================================

print("\nCommunity distribution:")

community_counts = (
    nlp_df["community"]
    .value_counts()
    .rename_axis("community")
    .reset_index(name="observations")
)

community_counts["percentage"] = (
    community_counts["observations"]
    / len(nlp_df)
    * 100
)

print(
    community_counts.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

community_counts.to_csv(
    TABLES_DIR / "nlp_corpus_community_counts.csv",
    index=False
)


# ============================================================
# YEAR COUNTS
# ============================================================

print("\nCommunity × year:")

community_year = (
    nlp_df
    .groupby(
        ["community", "year"]
    )
    .size()
    .reset_index(
        name="observations"
    )
)

print(
    community_year.to_string(
        index=False
    )
)

community_year.to_csv(
    TABLES_DIR / "nlp_corpus_community_year.csv",
    index=False
)


# ============================================================
# QUALITY SUMMARY
# ============================================================

quality_summary = pd.DataFrame({
    "feature": [
        "very_short",
        "long_text",
        "very_long_text",
        "article_like",
        "repeated_text",
        "sentence_boundary"
    ],
    "observations": [
        nlp_df["is_very_short"].sum(),
        nlp_df["is_long_text"].sum(),
        nlp_df["is_very_long_text"].sum(),
        nlp_df["is_article_like"].sum(),
        nlp_df["is_repeated_text"].sum(),
        nlp_df["has_sentence_boundary"].sum()
    ]
})

quality_summary["percentage"] = (
    quality_summary["observations"]
    / len(nlp_df)
    * 100
)

print("\nQuality summary:")

print(
    quality_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

quality_summary.to_csv(
    TABLES_DIR / "nlp_corpus_quality_summary.csv",
    index=False
)


# ============================================================
# SAVE NLP CORPUS
# ============================================================

nlp_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("NLP-READY CORPUS CREATED")
print("=" * 70)

print(
    f"\nOutput: {OUTPUT_FILE}"
)

print(
    f"Observations: {len(nlp_df):,}"
)

print(
    f"Columns: {len(nlp_df.columns)}"
)

print(
    "\nOriginal `text` column preserved."
)

print(
    "`analysis_text` contains only controlled "
    "whitespace normalization."
)

print(
    "\nNo stemming."
)

print(
    "No stop-word removal."
)

print(
    "No lowercasing."
)

print(
    "No punctuation stripping."
)

print(
    "No aggressive preprocessing."
)

print(
    "\nMaster corpus was NOT modified."
)

print("\n" + "=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)