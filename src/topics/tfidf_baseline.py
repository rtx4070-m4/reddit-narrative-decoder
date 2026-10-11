from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


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

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TF-IDF BASELINE ANALYSIS")
print("=" * 70)

print(f"\nInput: {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["created_utc"]
)

print(f"Observations: {len(df):,}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "record_id",
    "community",
    "analysis_text"
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
# TF-IDF CONFIGURATION
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",

    # Ignore extremely rare terms.
    min_df=2,

    # Ignore terms appearing in nearly all documents.
    max_df=0.95,

    # Unigrams + bigrams.
    ngram_range=(1, 2),

    # Remove standard English stopwords
    # ONLY for this TF-IDF baseline.
    stop_words="english",

    # Prevent extremely long vocabulary.
    max_features=10000,

    sublinear_tf=True
)


# ============================================================
# FIT TF-IDF
# ============================================================

print("\nFitting TF-IDF...")

X = vectorizer.fit_transform(
    df["analysis_text"]
)

feature_names = vectorizer.get_feature_names_out()

print(
    f"TF-IDF matrix shape: "
    f"{X.shape[0]:,} × {X.shape[1]:,}"
)

print(
    f"Non-zero values: "
    f"{X.nnz:,}"
)


# ============================================================
# GLOBAL TF-IDF TERMS
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL TF-IDF TERMS")
print("=" * 70)

mean_tfidf = np.asarray(
    X.mean(axis=0)
).ravel()

global_terms = pd.DataFrame({
    "term": feature_names,
    "mean_tfidf": mean_tfidf
})

global_terms = (
    global_terms
    .sort_values(
        "mean_tfidf",
        ascending=False
    )
    .reset_index(drop=True)
)

print(
    global_terms
    .head(50)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

global_terms.to_csv(
    TABLES_DIR / "tfidf_global_terms.csv",
    index=False
)


# ============================================================
# COMMUNITY-SPECIFIC TF-IDF
# ============================================================

print("\n" + "=" * 70)
print("COMMUNITY-SPECIFIC TF-IDF")
print("=" * 70)

community_results = []

for community in sorted(
    df["community"].unique()
):

    mask = (
        df["community"]
        .values
        == community
    )

    community_matrix = X[mask]

    community_mean = np.asarray(
        community_matrix.mean(axis=0)
    ).ravel()

    result = pd.DataFrame({
        "term": feature_names,
        "mean_tfidf": community_mean
    })

    result = (
        result
        .sort_values(
            "mean_tfidf",
            ascending=False
        )
        .head(50)
    )

    result.insert(
        0,
        "community",
        community
    )

    community_results.append(
        result
    )

    print(
        f"\n--- {community} ---"
    )

    print(
        result.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


community_terms = pd.concat(
    community_results,
    ignore_index=True
)

community_terms.to_csv(
    TABLES_DIR / "tfidf_community_terms.csv",
    index=False
)


# ============================================================
# COMMUNITY TERM MATRIX
# ============================================================

community_term_matrix = []

for community in sorted(
    df["community"].unique()
):

    mask = (
        df["community"]
        .values
        == community
    )

    community_mean = np.asarray(
        X[mask].mean(axis=0)
    ).ravel()

    community_term_matrix.append(
        community_mean
    )

community_term_matrix = np.vstack(
    community_term_matrix
)

community_names = sorted(
    df["community"].unique()
)

community_term_df = pd.DataFrame(
    community_term_matrix,
    index=community_names,
    columns=feature_names
)

community_term_df.to_csv(
    TABLES_DIR / "tfidf_community_term_matrix.csv"
)


# ============================================================
# DISCRIMINATIVE TERMS
# ============================================================

print("\n" + "=" * 70)
print("MOST DISCRIMINATIVE TERMS")
print("=" * 70)

discriminative_results = []

for i, community in enumerate(
    community_names
):

    own_scores = (
        community_term_matrix[i]
    )

    other_scores = (
        np.delete(
            community_term_matrix,
            i,
            axis=0
        )
        .mean(axis=0)
    )

    difference = (
        own_scores
        - other_scores
    )

    result = pd.DataFrame({
        "term": feature_names,
        "community_tfidf": own_scores,
        "other_communities_mean": other_scores,
        "difference": difference
    })

    result = (
        result
        .sort_values(
            "difference",
            ascending=False
        )
        .head(50)
    )

    result.insert(
        0,
        "community",
        community
    )

    discriminative_results.append(
        result
    )

    print(
        f"\n--- {community} ---"
    )

    print(
        result.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )


discriminative_terms = pd.concat(
    discriminative_results,
    ignore_index=True
)

discriminative_terms.to_csv(
    TABLES_DIR
    / "tfidf_discriminative_terms.csv",
    index=False
)


# ============================================================
# DOCUMENT-LEVEL METADATA
# ============================================================

tfidf_density = (
    X.getnnz(axis=1)
)

document_features = pd.DataFrame({
    "record_id": df["record_id"],
    "community": df["community"],
    "tfidf_nonzero_terms": tfidf_density
})

document_features.to_csv(
    TABLES_DIR
    / "tfidf_document_statistics.csv",
    index=False
)


# ============================================================
# SAVE VOCABULARY
# ============================================================

vocabulary = pd.DataFrame({
    "term": feature_names,
    "feature_index": np.arange(
        len(feature_names)
    )
})

vocabulary.to_csv(
    TABLES_DIR
    / "tfidf_vocabulary.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF BASELINE COMPLETE")
print("=" * 70)

print(
    f"\nDocuments: {X.shape[0]:,}"
)

print(
    f"Vocabulary: {X.shape[1]:,}"
)

print(
    f"Non-zero matrix values: {X.nnz:,}"
)

print("\nResults saved to:")
print(TABLES_DIR)

print("\nMaster/NLP corpus was NOT modified.")

print("\n" + "=" * 70)