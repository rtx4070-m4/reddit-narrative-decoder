from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CORPUS QUALITY PROFILING")
print("=" * 70)

print(f"\nInput: {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["created_utc"]
)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# BASIC STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATA STRUCTURE")
print("=" * 70)

print("\nColumns:")
for col in df.columns:
    print(f" - {col}")

print("\nData types:")
print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("2. MISSING VALUES")
print("=" * 70)

missing = (
    df.isna()
    .sum()
    .reset_index()
)

missing.columns = ["column", "missing_count"]

missing["missing_percent"] = (
    missing["missing_count"] / len(df) * 100
)

print(missing.to_string(index=False))

missing.to_csv(
    TABLES_DIR / "corpus_missing_values.csv",
    index=False
)


# ============================================================
# TEXT QUALITY FEATURES
# ============================================================

print("\n" + "=" * 70)
print("3. TEXT QUALITY FEATURES")
print("=" * 70)

df["text"] = df["text"].fillna("").astype(str)

df["char_count"] = df["text"].str.len()

df["word_count"] = (
    df["text"]
    .str.split()
    .str.len()
)

df["sentence_count"] = (
    df["text"]
    .str.count(r"[.!?]+")
)

df["uppercase_count"] = (
    df["text"]
    .str.count(r"[A-Z]")
)

df["digit_count"] = (
    df["text"]
    .str.count(r"\d")
)

df["punctuation_count"] = (
    df["text"]
    .str.count(r"[^\w\s]")
)

df["url_count"] = (
    df["text"]
    .str.lower()
    .str.count(r"https?://|www\.")
)

df["uppercase_ratio"] = np.where(
    df["char_count"] > 0,
    df["uppercase_count"] / df["char_count"],
    0
)

df["digit_ratio"] = np.where(
    df["char_count"] > 0,
    df["digit_count"] / df["char_count"],
    0
)

df["punctuation_ratio"] = np.where(
    df["char_count"] > 0,
    df["punctuation_count"] / df["char_count"],
    0
)


# ============================================================
# OVERALL TEXT STATISTICS
# ============================================================

text_stats = pd.DataFrame({
    "metric": [
        "observations",
        "mean_characters",
        "median_characters",
        "minimum_characters",
        "maximum_characters",
        "mean_words",
        "median_words",
        "minimum_words",
        "maximum_words",
        "mean_sentences",
        "median_sentences",
        "texts_under_3_chars",
        "texts_under_10_words",
        "texts_over_250_words",
        "texts_over_500_words",
        "texts_over_1000_words",
        "texts_with_urls",
    ],
    "value": [
        len(df),
        df["char_count"].mean(),
        df["char_count"].median(),
        df["char_count"].min(),
        df["char_count"].max(),
        df["word_count"].mean(),
        df["word_count"].median(),
        df["word_count"].min(),
        df["word_count"].max(),
        df["sentence_count"].mean(),
        df["sentence_count"].median(),
        (df["char_count"] < 3).sum(),
        (df["word_count"] < 10).sum(),
        (df["word_count"] > 250).sum(),
        (df["word_count"] > 500).sum(),
        (df["word_count"] > 1000).sum(),
        (df["url_count"] > 0).sum(),
    ]
})

print("\nOverall text statistics:")
print(text_stats.to_string(index=False))

text_stats.to_csv(
    TABLES_DIR / "corpus_text_statistics.csv",
    index=False
)


# ============================================================
# COMMUNITY PROFILE
# ============================================================

print("\n" + "=" * 70)
print("4. COMMUNITY PROFILE")
print("=" * 70)

community_profile = (
    df.groupby("community")
    .agg(
        observations=("record_id", "count"),
        mean_characters=("char_count", "mean"),
        median_characters=("char_count", "median"),
        mean_words=("word_count", "mean"),
        median_words=("word_count", "median"),
        min_words=("word_count", "min"),
        max_words=("word_count", "max"),
        texts_with_urls=("url_count", lambda x: (x > 0).sum()),
    )
    .reset_index()
)

community_profile["percentage"] = (
    community_profile["observations"]
    / len(df)
    * 100
)

community_profile = community_profile[
    [
        "community",
        "observations",
        "percentage",
        "mean_characters",
        "median_characters",
        "mean_words",
        "median_words",
        "min_words",
        "max_words",
        "texts_with_urls",
    ]
]

print(
    community_profile.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

community_profile.to_csv(
    TABLES_DIR / "community_profile.csv",
    index=False
)


# ============================================================
# TEMPORAL PROFILE
# ============================================================

print("\n" + "=" * 70)
print("5. TEMPORAL PROFILE")
print("=" * 70)

df["year"] = df["created_utc"].dt.year

df["month_period"] = (
    df["created_utc"]
    .dt.to_period("M")
    .astype(str)
)

year_profile = (
    df.groupby(["community", "year"])
    .size()
    .reset_index(name="observations")
)

year_profile["percentage_within_community"] = (
    year_profile
    .groupby("community")["observations"]
    .transform(lambda x: x / x.sum() * 100)
)

print("\nCommunity × Year:")
print(
    year_profile.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

year_profile.to_csv(
    TABLES_DIR / "community_year_profile.csv",
    index=False
)


month_profile = (
    df.groupby(["community", "month_period"])
    .size()
    .reset_index(name="observations")
)

month_profile.to_csv(
    TABLES_DIR / "community_month_profile.csv",
    index=False
)


# ============================================================
# DUPLICATE TEXT PROFILE
# ============================================================

print("\n" + "=" * 70)
print("6. DUPLICATE TEXT PROFILE")
print("=" * 70)

unique_texts = df["text"].nunique()

duplicate_text_rows = (
    len(df) - unique_texts
)

print(f"Total observations: {len(df):,}")
print(f"Unique texts: {unique_texts:,}")
print(f"Repeated-text rows: {duplicate_text_rows:,}")

text_frequency = (
    df.groupby("text")
    .size()
    .reset_index(name="frequency")
    .sort_values("frequency", ascending=False)
)

print("\nTop repeated texts:")

print(
    text_frequency.head(10).to_string(
        index=False,
        max_colwidth=100
    )
)

text_frequency.head(100).to_csv(
    TABLES_DIR / "top_repeated_texts.csv",
    index=False
)


# ============================================================
# TEXT LENGTH DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("7. TEXT LENGTH DISTRIBUTION")
print("=" * 70)

bins = [
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    np.inf
]

labels = [
    "1–10",
    "11–25",
    "26–50",
    "51–100",
    "101–250",
    "251–500",
    "501–1000",
    "1000+"
]

df["word_length_group"] = pd.cut(
    df["word_count"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

length_distribution = (
    df["word_length_group"]
    .value_counts(sort=False)
    .reset_index()
)

length_distribution.columns = [
    "word_count_group",
    "observations"
]

length_distribution["percentage"] = (
    length_distribution["observations"]
    / len(df)
    * 100
)

print(
    length_distribution.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

length_distribution.to_csv(
    TABLES_DIR / "text_length_distribution.csv",
    index=False
)


# ============================================================
# VERY SHORT / VERY LONG TEXTS
# ============================================================

print("\n" + "=" * 70)
print("8. EXTREME TEXT LENGTHS")
print("=" * 70)

short_texts = (
    df.sort_values("word_count")
    [
        [
            "record_id",
            "community",
            "created_utc",
            "word_count",
            "char_count",
            "text"
        ]
    ]
    .head(20)
)

long_texts = (
    df.sort_values(
        "word_count",
        ascending=False
    )
    [
        [
            "record_id",
            "community",
            "created_utc",
            "word_count",
            "char_count",
            "text"
        ]
    ]
    .head(20)
)

short_texts.to_csv(
    TABLES_DIR / "shortest_texts.csv",
    index=False
)

long_texts.to_csv(
    TABLES_DIR / "longest_texts.csv",
    index=False
)

print("\nShortest texts:")
print(
    short_texts[
        [
            "record_id",
            "community",
            "word_count",
            "text"
        ]
    ].to_string(index=False, max_colwidth=80)
)

print("\nLongest texts:")
print(
    long_texts[
        [
            "record_id",
            "community",
            "word_count",
            "text"
        ]
    ].to_string(index=False, max_colwidth=80)
)


# ============================================================
# FIGURE 1 — COMMUNITY SIZE
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    community_profile["community"],
    community_profile["observations"]
)

plt.xlabel("Community")
plt.ylabel("Number of observations")
plt.title("Observations by Reddit Community")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "community_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 2 — WORD COUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    df["word_count"],
    bins=50
)

plt.xlabel("Word count")
plt.ylabel("Frequency")
plt.title("Distribution of Text Length")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "word_count_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 3 — COMMUNITY × YEAR
# ============================================================

pivot_year = (
    year_profile
    .pivot(
        index="year",
        columns="community",
        values="observations"
    )
    .fillna(0)
)

pivot_year.plot(
    figsize=(10, 6),
    marker="o"
)

plt.xlabel("Year")
plt.ylabel("Number of observations")
plt.title("Temporal Coverage by Community")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "community_year_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE PROFILE DATASET
# ============================================================

profile_columns = [
    "record_id",
    "id",
    "community",
    "created_utc",
    "text",
    "char_count",
    "word_count",
    "sentence_count",
    "uppercase_count",
    "digit_count",
    "punctuation_count",
    "url_count",
    "uppercase_ratio",
    "digit_ratio",
    "punctuation_ratio",
]

df[profile_columns].to_csv(
    TABLES_DIR / "corpus_profile_records.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CORPUS PROFILING COMPLETE")
print("=" * 70)

print(f"\nObservations: {len(df):,}")
print(f"Unique texts: {unique_texts:,}")
print(f"Communities: {df['community'].nunique()}")
print(
    f"Date range: "
    f"{df['created_utc'].min()} → "
    f"{df['created_utc'].max()}"
)

print("\nTables saved to:")
print(TABLES_DIR)

print("\nFigures saved to:")
print(FIGURES_DIR)

print("\n" + "=" * 70)
print("NEXT STEP: REVIEW THE CORPUS PROFILE")
print("=" * 70)