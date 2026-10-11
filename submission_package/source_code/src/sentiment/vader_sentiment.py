from pathlib import Path
import sys

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_FILE = (
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("VADER SENTIMENT ANALYSIS")
print("=" * 70)

print(f"\nInput: {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["created_utc"]
)

print(
    f"Observations: {len(df):,}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "record_id",
    "community",
    "analysis_text",
    "created_utc",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df["analysis_text"].isna().any():
    raise ValueError(
        "analysis_text contains missing values."
    )


# ============================================================
# INITIALIZE VADER
# ============================================================

analyzer = SentimentIntensityAnalyzer()


# ============================================================
# SENTIMENT FUNCTION
# ============================================================

def calculate_sentiment(text):

    scores = analyzer.polarity_scores(
        str(text)
    )

    return pd.Series({
        "vader_negative": scores["neg"],
        "vader_neutral": scores["neu"],
        "vader_positive": scores["pos"],
        "vader_compound": scores["compound"],
    })


# ============================================================
# CALCULATE SENTIMENT
# ============================================================

print("\nCalculating VADER sentiment...")

sentiment_scores = (
    df["analysis_text"]
    .apply(calculate_sentiment)
)

df = pd.concat(
    [
        df,
        sentiment_scores
    ],
    axis=1
)


# ============================================================
# SENTIMENT LABEL
# ============================================================

def classify_sentiment(compound):

    if compound >= 0.05:
        return "positive"

    elif compound <= -0.05:
        return "negative"

    else:
        return "neutral"


df["vader_label"] = (
    df["vader_compound"]
    .apply(classify_sentiment)
)


# ============================================================
# SENTIMENT INTENSITY
# ============================================================

df["vader_abs_compound"] = (
    df["vader_compound"]
    .abs()
)

df["vader_emotional_intensity"] = (
    df[
        [
            "vader_negative",
            "vader_positive"
        ]
    ]
    .sum(axis=1)
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SENTIMENT VALIDATION")
print("=" * 70)

print(
    "\nMissing sentiment scores:"
)

print(
    df[
        [
            "vader_negative",
            "vader_neutral",
            "vader_positive",
            "vader_compound",
            "vader_label"
        ]
    ]
    .isna()
    .sum()
    .to_string()
)


print(
    f"\nScore range:"
)

print(
    f"Compound minimum: "
    f"{df['vader_compound'].min():.4f}"
)

print(
    f"Compound maximum: "
    f"{df['vader_compound'].max():.4f}"
)


# ============================================================
# OVERALL SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("OVERALL SENTIMENT DISTRIBUTION")
print("=" * 70)

overall_counts = (
    df["vader_label"]
    .value_counts()
    .rename_axis("sentiment")
    .reset_index(name="observations")
)

overall_counts["percentage"] = (
    overall_counts["observations"]
    / len(df)
    * 100
)

print(
    overall_counts.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

overall_counts.to_csv(
    TABLES_DIR
    / "vader_overall_distribution.csv",
    index=False
)


# ============================================================
# COMMUNITY SENTIMENT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMMUNITY SENTIMENT SUMMARY")
print("=" * 70)

community_summary = (
    df.groupby("community")
    .agg(
        observations=("record_id", "count"),

        mean_negative=(
            "vader_negative",
            "mean"
        ),

        mean_neutral=(
            "vader_neutral",
            "mean"
        ),

        mean_positive=(
            "vader_positive",
            "mean"
        ),

        mean_compound=(
            "vader_compound",
            "mean"
        ),

        median_compound=(
            "vader_compound",
            "median"
        ),

        mean_intensity=(
            "vader_emotional_intensity",
            "mean"
        ),
    )
    .reset_index()
)

print(
    community_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

community_summary.to_csv(
    TABLES_DIR
    / "vader_community_summary.csv",
    index=False
)


# ============================================================
# COMMUNITY × SENTIMENT LABEL
# ============================================================

community_label_counts = (
    pd.crosstab(
        df["community"],
        df["vader_label"]
    )
    .reset_index()
)

print(
    "\nCommunity × sentiment:"
)

print(
    community_label_counts.to_string(
        index=False
    )
)

community_label_counts.to_csv(
    TABLES_DIR
    / "vader_community_label_counts.csv",
    index=False
)


# ============================================================
# COMMUNITY × SENTIMENT PERCENTAGES
# ============================================================

community_label_percent = (
    pd.crosstab(
        df["community"],
        df["vader_label"],
        normalize="index"
    )
    * 100
)

community_label_percent = (
    community_label_percent
    .reset_index()
)

print(
    "\nCommunity × sentiment percentages:"
)

print(
    community_label_percent.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

community_label_percent.to_csv(
    TABLES_DIR
    / "vader_community_label_percentages.csv",
    index=False
)


# ============================================================
# YEAR × SENTIMENT
# ============================================================

df["year"] = (
    df["created_utc"]
    .dt.year
)

year_summary = (
    df.groupby("year")
    .agg(
        observations=("record_id", "count"),
        mean_negative=(
            "vader_negative",
            "mean"
        ),
        mean_neutral=(
            "vader_neutral",
            "mean"
        ),
        mean_positive=(
            "vader_positive",
            "mean"
        ),
        mean_compound=(
            "vader_compound",
            "mean"
        ),
    )
    .reset_index()
)

print(
    "\nYear-level sentiment:"
)

print(
    year_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

year_summary.to_csv(
    TABLES_DIR
    / "vader_year_summary.csv",
    index=False
)


# ============================================================
# COMMUNITY × YEAR SENTIMENT
# ============================================================

community_year_summary = (
    df.groupby(
        [
            "community",
            "year"
        ]
    )
    .agg(
        observations=("record_id", "count"),

        mean_negative=(
            "vader_negative",
            "mean"
        ),

        mean_neutral=(
            "vader_neutral",
            "mean"
        ),

        mean_positive=(
            "vader_positive",
            "mean"
        ),

        mean_compound=(
            "vader_compound",
            "mean"
        ),

        median_compound=(
            "vader_compound",
            "median"
        ),
    )
    .reset_index()
)

print(
    "\nCommunity × year sentiment:"
)

print(
    community_year_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

community_year_summary.to_csv(
    TABLES_DIR
    / "vader_community_year_summary.csv",
    index=False
)


# ============================================================
# EXTREME SENTIMENT EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("EXTREME SENTIMENT OBSERVATIONS")
print("=" * 70)


most_negative = (
    df[
        [
            "record_id",
            "community",
            "created_utc",
            "vader_compound",
            "vader_negative",
            "vader_positive",
            "analysis_text"
        ]
    ]
    .sort_values(
        "vader_compound"
    )
    .head(20)
)

most_positive = (
    df[
        [
            "record_id",
            "community",
            "created_utc",
            "vader_compound",
            "vader_negative",
            "vader_positive",
            "analysis_text"
        ]
    ]
    .sort_values(
        "vader_compound",
        ascending=False
    )
    .head(20)
)

print("\nMost negative:")
print(
    most_negative.to_string(
        index=False,
        max_colwidth=100
    )
)

print("\nMost positive:")
print(
    most_positive.to_string(
        index=False,
        max_colwidth=100
    )
)

most_negative.to_csv(
    TABLES_DIR
    / "vader_most_negative.csv",
    index=False
)

most_positive.to_csv(
    TABLES_DIR
    / "vader_most_positive.csv",
    index=False
)


# ============================================================
# SAVE COMPLETE SENTIMENT DATASET
# ============================================================

output_columns = [
    "record_id",
    "id",
    "community",
    "subreddit",
    "created_utc",
    "year",
    "text",
    "analysis_text",

    "vader_negative",
    "vader_neutral",
    "vader_positive",
    "vader_compound",
    "vader_label",
    "vader_abs_compound",
    "vader_emotional_intensity",

    "is_very_short",
    "is_article_like",
    "is_repeated_text",
]


sentiment_output = df[
    output_columns
]

sentiment_file = (
    TABLES_DIR
    / "vader_sentiment_results.csv"
)

sentiment_output.to_csv(
    sentiment_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VADER SENTIMENT ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nObservations analyzed: "
    f"{len(df):,}"
)

print(
    f"Communities: "
    f"{df['community'].nunique()}"
)

print(
    f"\nComplete results:"
)

print(
    sentiment_file
)

print(
    "\nCommunity summary:"
)

print(
    TABLES_DIR
    / "vader_community_summary.csv"
)

print(
    "\nMaster/NLP corpus was NOT modified."
)

print("\n" + "=" * 70)