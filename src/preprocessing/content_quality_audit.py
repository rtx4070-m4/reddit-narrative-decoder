from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np


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

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CONTENT QUALITY AUDIT")
print("=" * 70)

print(f"\nInput: {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["created_utc"]
)

df["text"] = df["text"].fillna("").astype(str)

print(f"Observations: {len(df):,}")


# ============================================================
# BASIC TEXT FEATURES
# ============================================================

df["char_count"] = df["text"].str.len()

df["word_count"] = (
    df["text"]
    .str.split()
    .str.len()
)


# ============================================================
# SENTENCE DETECTION
# ============================================================

# Count sentence-ending punctuation.
df["sentence_boundary_count"] = df["text"].apply(
    lambda x: len(
        re.findall(
            r"[.!?]+",
            x
        )
    )
)

df["has_sentence_boundary"] = (
    df["sentence_boundary_count"] > 0
)


# ============================================================
# URL DETECTION
# ============================================================

URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+)",
    flags=re.IGNORECASE
)

df["url_count"] = df["text"].apply(
    lambda x: len(URL_PATTERN.findall(x))
)

df["has_url"] = df["url_count"] > 0


# ============================================================
# PLACEHOLDER / REMOVED CONTENT
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
    normalized_text.isin(PLACEHOLDER_VALUES)
)


# ============================================================
# VERY SHORT TEXT
# ============================================================

df["is_very_short"] = (
    df["word_count"] <= 3
)


# ============================================================
# LONG TEXT
# ============================================================

df["is_long_text"] = (
    df["word_count"] > 250
)

df["is_very_long_text"] = (
    df["word_count"] > 500
)


# ============================================================
# ARTICLE-LIKE CONTENT
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
    r"\bwww\.",
]

article_regex = re.compile(
    "|".join(ARTICLE_PATTERNS),
    flags=re.IGNORECASE
)

df["is_article_like"] = (
    df["text"].apply(
        lambda x: bool(article_regex.search(x))
    )
)


# ============================================================
# REPETITION
# ============================================================

text_frequency = (
    df.groupby("text")["record_id"]
    .transform("count")
)

df["text_frequency"] = text_frequency

df["is_repeated_text"] = (
    df["text_frequency"] > 1
)


# ============================================================
# SPECIAL CHARACTER / PUNCTUATION SIGNALS
# ============================================================

df["punctuation_count"] = df["text"].apply(
    lambda x: len(
        re.findall(
            r"[^\w\s]",
            x,
            flags=re.UNICODE
        )
    )
)

df["has_punctuation"] = (
    df["punctuation_count"] > 0
)


# ============================================================
# UPPERCASE SIGNAL
# ============================================================

df["uppercase_count"] = df["text"].apply(
    lambda x: len(
        re.findall(
            r"[A-Z]",
            x
        )
    )
)

df["has_uppercase"] = (
    df["uppercase_count"] > 0
)


# ============================================================
# DIGIT SIGNAL
# ============================================================

df["digit_count"] = df["text"].apply(
    lambda x: len(
        re.findall(
            r"\d",
            x
        )
    )
)

df["has_digits"] = (
    df["digit_count"] > 0
)


# ============================================================
# QUALITY CATEGORY
# ============================================================

def classify_quality(row):

    if row["is_placeholder"]:
        return "placeholder"

    if row["is_article_like"]:
        return "article_like"

    if row["is_very_long_text"]:
        return "very_long"

    if row["is_long_text"]:
        return "long"

    if row["is_very_short"]:
        return "very_short"

    return "normal"


df["primary_quality_category"] = df.apply(
    classify_quality,
    axis=1
)


# ============================================================
# PRINT QUALITY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("QUALITY FLAGS")
print("=" * 70)

quality_summary = pd.DataFrame({
    "quality_flag": [
        "placeholder",
        "very_short",
        "long_text",
        "very_long_text",
        "article_like",
        "repeated_text",
        "has_url",
        "has_sentence_boundary",
        "has_punctuation",
        "has_uppercase",
        "has_digits",
    ],
    "observations": [
        df["is_placeholder"].sum(),
        df["is_very_short"].sum(),
        df["is_long_text"].sum(),
        df["is_very_long_text"].sum(),
        df["is_article_like"].sum(),
        df["is_repeated_text"].sum(),
        df["has_url"].sum(),
        df["has_sentence_boundary"].sum(),
        df["has_punctuation"].sum(),
        df["has_uppercase"].sum(),
        df["has_digits"].sum(),
    ]
})

quality_summary["percentage"] = (
    quality_summary["observations"]
    / len(df)
    * 100
)

print(
    quality_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

quality_summary.to_csv(
    TABLES_DIR / "content_quality_summary.csv",
    index=False
)


# ============================================================
# PRIMARY QUALITY CATEGORY
# ============================================================

print("\n" + "=" * 70)
print("PRIMARY QUALITY CATEGORY")
print("=" * 70)

category_summary = (
    df["primary_quality_category"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="observations")
)

category_summary["percentage"] = (
    category_summary["observations"]
    / len(df)
    * 100
)

print(
    category_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

category_summary.to_csv(
    TABLES_DIR / "primary_quality_categories.csv",
    index=False
)


# ============================================================
# COMMUNITY × QUALITY
# ============================================================

print("\n" + "=" * 70)
print("COMMUNITY × QUALITY")
print("=" * 70)

community_quality = (
    pd.crosstab(
        df["community"],
        df["primary_quality_category"]
    )
    .reset_index()
)

print(
    community_quality.to_string(
        index=False
    )
)

community_quality.to_csv(
    TABLES_DIR / "community_quality_categories.csv",
    index=False
)


# ============================================================
# PLACEHOLDER TEXTS
# ============================================================

print("\n" + "=" * 70)
print("PLACEHOLDER TEXTS")
print("=" * 70)

placeholders = (
    df[df["is_placeholder"]]
    [
        [
            "record_id",
            "community",
            "created_utc",
            "word_count",
            "text"
        ]
    ]
    .sort_values(
        ["community", "text"]
    )
)

print(
    placeholders.to_string(
        index=False
    )
)

placeholders.to_csv(
    TABLES_DIR / "placeholder_texts.csv",
    index=False
)


# ============================================================
# ARTICLE-LIKE TEXTS
# ============================================================

print("\n" + "=" * 70)
print("ARTICLE-LIKE TEXTS")
print("=" * 70)

article_like = (
    df[df["is_article_like"]]
    [
        [
            "record_id",
            "community",
            "created_utc",
            "word_count",
            "text"
        ]
    ]
    .sort_values(
        "word_count",
        ascending=False
    )
)

print(
    f"Article-like observations: {len(article_like):,}"
)

print(
    article_like[
        [
            "record_id",
            "community",
            "word_count",
            "text"
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        max_colwidth=100
    )
)

article_like.to_csv(
    TABLES_DIR / "article_like_texts.csv",
    index=False
)


# ============================================================
# VERY SHORT TEXTS
# ============================================================

print("\n" + "=" * 70)
print("VERY SHORT TEXTS")
print("=" * 70)

very_short = (
    df[df["is_very_short"]]
    [
        [
            "record_id",
            "community",
            "created_utc",
            "word_count",
            "text"
        ]
    ]
)

print(
    f"Very-short observations: {len(very_short):,}"
)

very_short.to_csv(
    TABLES_DIR / "very_short_texts.csv",
    index=False
)


# ============================================================
# VERY LONG TEXTS
# ============================================================

print("\n" + "=" * 70)
print("VERY LONG TEXTS")
print("=" * 70)

very_long = (
    df[df["is_very_long_text"]]
    [
        [
            "record_id",
            "community",
            "created_utc",
            "word_count",
            "text"
        ]
    ]
    .sort_values(
        "word_count",
        ascending=False
    )
)

print(
    f"Very-long observations: {len(very_long):,}"
)

print(
    very_long[
        [
            "record_id",
            "community",
            "word_count",
            "text"
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        max_colwidth=100
    )
)

very_long.to_csv(
    TABLES_DIR / "very_long_texts.csv",
    index=False
)


# ============================================================
# URL-CONTAINING TEXTS
# ============================================================

print("\n" + "=" * 70)
print("URL-CONTAINING TEXTS")
print("=" * 70)

url_texts = (
    df[df["has_url"]]
    [
        [
            "record_id",
            "community",
            "created_utc",
            "url_count",
            "text"
        ]
    ]
)

print(
    f"Observations containing URLs: {len(url_texts):,}"
)

url_texts.to_csv(
    TABLES_DIR / "url_containing_texts.csv",
    index=False
)


# ============================================================
# COMPLETE AUDIT DATASET
# ============================================================

audit_columns = [
    "record_id",
    "id",
    "community",
    "subreddit",
    "created_utc",
    "text",
    "char_count",
    "word_count",
    "sentence_boundary_count",
    "url_count",
    "punctuation_count",
    "uppercase_count",
    "digit_count",
    "text_frequency",
    "is_placeholder",
    "is_very_short",
    "is_long_text",
    "is_very_long_text",
    "is_article_like",
    "is_repeated_text",
    "has_url",
    "has_sentence_boundary",
    "has_punctuation",
    "has_uppercase",
    "has_digits",
    "primary_quality_category",
]

df[audit_columns].to_csv(
    TABLES_DIR / "content_quality_audit.csv",
    index=False
)


# ============================================================
# FINAL RECOMMENDATION COUNTS
# ============================================================

print("\n" + "=" * 70)
print("POTENTIAL ANALYTICAL FLAGS")
print("=" * 70)

print(
    "\nThese are FLAGS only. No observations are being deleted."
)

print(
    f"Placeholder:       {df['is_placeholder'].sum():,}"
)

print(
    f"Very short:        {df['is_very_short'].sum():,}"
)

print(
    f"Article-like:      {df['is_article_like'].sum():,}"
)

print(
    f"Very long:         {df['is_very_long_text'].sum():,}"
)

print(
    f"Repeated text:     {df['is_repeated_text'].sum():,}"
)

print(
    f"Contains URL:      {df['has_url'].sum():,}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("CONTENT QUALITY AUDIT COMPLETE")
print("=" * 70)

print("\nMaster corpus was NOT modified.")

print("\nAudit tables saved to:")
print(TABLES_DIR)

print("\nMain audit file:")
print(
    TABLES_DIR / "content_quality_audit.csv"
)

print("\n" + "=" * 70)