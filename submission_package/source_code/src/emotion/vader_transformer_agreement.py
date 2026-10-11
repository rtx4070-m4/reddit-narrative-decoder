from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "transformer_emotion_results.csv"
)

VADER_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "vader_sentiment_results.csv"
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
# CONFIGURATION
# ============================================================

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

SENTIMENTS = [
    "negative",
    "neutral",
    "positive",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("VADER ↔ TRANSFORMER AGREEMENT ANALYSIS")
print("=" * 70)

transformer = pd.read_csv(
    INPUT_FILE
)

vader = pd.read_csv(
    VADER_FILE
)


print(
    f"\nTransformer observations: "
    f"{len(transformer):,}"
)

print(
    f"VADER observations: "
    f"{len(vader):,}"
)


# ============================================================
# SELECT REQUIRED VADER COLUMNS
# ============================================================

vader_required = [
    "record_id",
    "vader_compound",
    "vader_label",
]

missing_vader = [
    c for c in vader_required
    if c not in vader.columns
]

if missing_vader:

    raise ValueError(
        f"Missing VADER columns: "
        f"{missing_vader}"
    )


# ============================================================
# SELECT REQUIRED TRANSFORMER COLUMNS
# ============================================================

transformer_required = [
    "record_id",
    "community",
    "created_utc",
    "emotion_label",
    "emotion_confidence",
]

missing_transformer = [
    c for c in transformer_required
    if c not in transformer.columns
]

if missing_transformer:

    raise ValueError(
        f"Missing transformer columns: "
        f"{missing_transformer}"
    )


# ============================================================
# MERGE
# ============================================================

df = transformer[
    transformer_required
].merge(
    vader[
        vader_required
    ],
    on="record_id",
    how="inner",
    validate="one_to_one",
)


print(
    f"\nMatched observations: "
    f"{len(df):,}"
)


if len(df) != len(transformer):

    raise ValueError(
        "Not all transformer observations "
        "matched the VADER results."
    )


# ============================================================
# VALIDATION
# ============================================================

if df["vader_label"].isna().any():

    raise ValueError(
        "Missing VADER labels."
    )

if df["emotion_label"].isna().any():

    raise ValueError(
        "Missing transformer emotion labels."
    )


# ============================================================
# PART A
# VADER × TRANSFORMER CROSS-TABULATION
# ============================================================

print("\n" + "=" * 70)
print("PART A — SENTIMENT × EMOTION")
print("=" * 70)

cross_tab = pd.crosstab(
    df["vader_label"],
    df["emotion_label"],
)

cross_tab = cross_tab.reindex(
    index=SENTIMENTS,
    columns=EMOTIONS,
    fill_value=0,
)

print(
    "\nCounts:"
)

print(
    cross_tab.to_string()
)

cross_tab.to_csv(
    TABLES_DIR
    / "vader_transformer_crosstab_counts.csv"
)


# ============================================================
# ROW PERCENTAGES
# ============================================================

row_percent = (
    cross_tab
    .div(
        cross_tab.sum(axis=1),
        axis=0
    )
    * 100
)

print(
    "\nRow percentages:"
)

print(
    row_percent.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

row_percent.to_csv(
    TABLES_DIR
    / "vader_transformer_crosstab_row_percent.csv"
)


# ============================================================
# COLUMN PERCENTAGES
# ============================================================

column_percent = (
    cross_tab
    .div(
        cross_tab.sum(axis=0),
        axis=1
    )
    * 100
)

print(
    "\nColumn percentages:"
)

print(
    column_percent.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

column_percent.to_csv(
    TABLES_DIR
    / "vader_transformer_crosstab_column_percent.csv"
)


# ============================================================
# PART B
# CHI-SQUARE ASSOCIATION
# ============================================================

print("\n" + "=" * 70)
print("PART B — ASSOCIATION TEST")
print("=" * 70)

chi2, p_value, df_chi, expected = (
    chi2_contingency(
        cross_tab
    )
)

print(
    f"\nChi-square: "
    f"{chi2:.4f}"
)

print(
    f"Degrees of freedom: "
    f"{df_chi}"
)

print(
    f"p-value: "
    f"{p_value:.10g}"
)


n = cross_tab.to_numpy().sum()

r, c = cross_tab.shape

cramers_v = np.sqrt(
    chi2
    / (
        n
        * min(
            r - 1,
            c - 1
        )
    )
)

print(
    f"Cramer's V: "
    f"{cramers_v:.6f}"
)


pd.DataFrame({
    "test": ["VADER_vs_transformer"],
    "chi2": [chi2],
    "df": [df_chi],
    "p_value": [p_value],
    "cramers_v": [cramers_v],
    "N": [n],
}).to_csv(
    TABLES_DIR
    / "vader_transformer_association.csv",
    index=False
)


# ============================================================
# PART C
# COHEN'S KAPPA — ONLY FOR SHARED "NEUTRAL"
# ============================================================

print("\n" + "=" * 70)
print("PART C — SHARED NEUTRAL AGREEMENT")
print("=" * 70)

"""
Important:
VADER has 3 sentiment classes while the transformer
has 7 emotion classes.

Therefore, a standard Cohen's kappa across the
original labels is NOT scientifically appropriate.

The only directly shared label is "neutral".

We therefore report neutral classification agreement
descriptively rather than pretending the two models
have equivalent class systems.
"""

vader_neutral = (
    df["vader_label"] == "neutral"
)

transformer_neutral = (
    df["emotion_label"] == "neutral"
)

neutral_agreement = (
    vader_neutral
    == transformer_neutral
)

print(
    f"\nBoth neutral: "
    f"{(vader_neutral & transformer_neutral).sum():,}"
)

print(
    f"VADER neutral / transformer non-neutral: "
    f"{(vader_neutral & ~transformer_neutral).sum():,}"
)

print(
    f"VADER non-neutral / transformer neutral: "
    f"{(~vader_neutral & transformer_neutral).sum():,}"
)

print(
    f"Both non-neutral: "
    f"{(~vader_neutral & ~transformer_neutral).sum():,}"
)

print(
    f"\nOverall binary neutral agreement: "
    f"{neutral_agreement.mean() * 100:.2f}%"
)

neutral_kappa = cohen_kappa_score(
    vader_neutral,
    transformer_neutral
)

print(
    f"Binary neutral Cohen's kappa: "
    f"{neutral_kappa:.4f}"
)


pd.DataFrame({
    "metric": [
        "both_neutral",
        "vader_neutral_transformer_non_neutral",
        "vader_non_neutral_transformer_neutral",
        "both_non_neutral",
        "neutral_binary_agreement",
        "neutral_binary_kappa",
    ],
    "value": [
        (
            vader_neutral
            & transformer_neutral
        ).sum(),

        (
            vader_neutral
            & ~transformer_neutral
        ).sum(),

        (
            ~vader_neutral
            & transformer_neutral
        ).sum(),

        (
            ~vader_neutral
            & ~transformer_neutral
        ).sum(),

        neutral_agreement.mean(),

        neutral_kappa,
    ],
}).to_csv(
    TABLES_DIR
    / "vader_transformer_neutral_agreement.csv",
    index=False
)


# ============================================================
# PART D
# EMOTION-SPECIFIC VADER COMPOUND SCORES
# ============================================================

print("\n" + "=" * 70)
print("PART D — VADER SENTIMENT WITHIN EACH EMOTION")
print("=" * 70)

emotion_vader_summary = (
    df.groupby(
        "emotion_label",
        observed=True
    )
    .agg(
        observations=(
            "record_id",
            "count"
        ),

        mean_vader_compound=(
            "vader_compound",
            "mean"
        ),

        median_vader_compound=(
            "vader_compound",
            "median"
        ),

        mean_vader_negative=(
            "vader_compound",
            lambda x: np.nan
        ),
    )
    .reset_index()
)

# Remove placeholder aggregate column
emotion_vader_summary = (
    emotion_vader_summary.drop(
        columns=["mean_vader_negative"]
    )
)

print(
    emotion_vader_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

emotion_vader_summary.to_csv(
    TABLES_DIR
    / "emotion_vader_sentiment_summary.csv",
    index=False
)


# ============================================================
# PART E
# SENTIMENT DISTRIBUTION WITHIN EACH EMOTION
# ============================================================

emotion_sentiment_percent = (
    pd.crosstab(
        df["emotion_label"],
        df["vader_label"],
        normalize="index"
    )
    * 100
)

emotion_sentiment_percent = (
    emotion_sentiment_percent
    .reindex(
        index=EMOTIONS,
        columns=SENTIMENTS,
        fill_value=0
    )
)

print(
    "\nVADER sentiment percentages "
    "within each transformer emotion:"
)

print(
    emotion_sentiment_percent.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)

emotion_sentiment_percent.to_csv(
    TABLES_DIR
    / "emotion_vader_sentiment_percentages.csv"
)


# ============================================================
# PART F
# COMMUNITY-SPECIFIC AGREEMENT
# ============================================================

print("\n" + "=" * 70)
print("PART F — COMMUNITY-SPECIFIC AGREEMENT")
print("=" * 70)

community_results = []

for community in sorted(
    df["community"].dropna().unique()
):

    subset = df[
        df["community"] == community
    ]

    v_neutral = (
        subset["vader_label"]
        == "neutral"
    )

    t_neutral = (
        subset["emotion_label"]
        == "neutral"
    )

    agreement = (
        v_neutral
        == t_neutral
    )

    kappa = cohen_kappa_score(
        v_neutral,
        t_neutral
    )

    community_results.append({
        "community": community,
        "observations": len(subset),
        "neutral_agreement": (
            agreement.mean()
        ),
        "neutral_kappa": kappa,
        "both_neutral": (
            v_neutral
            & t_neutral
        ).sum(),
    })


community_results = pd.DataFrame(
    community_results
)

print(
    community_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

community_results.to_csv(
    TABLES_DIR
    / "vader_transformer_community_agreement.csv",
    index=False
)


# ============================================================
# PART G
# HIGH DISAGREEMENT / INTERESTING CASES
# ============================================================

print("\n" + "=" * 70)
print("PART G — HIGH VADER/TRANSFORMER DIVERGENCE")
print("=" * 70)

"""
A simple divergence score:

VADER absolute compound intensity
vs
transformer confidence.

This is NOT a disagreement truth score.
It simply identifies observations worth
qualitative inspection.
"""

df["vader_intensity"] = (
    df["vader_compound"].abs()
)

df["divergence_score"] = (
    df["vader_intensity"]
    * df["emotion_confidence"]
)

interesting = (
    df[
        [
            "record_id",
            "community",
            "vader_label",
            "vader_compound",
            "emotion_label",
            "emotion_confidence",
            "divergence_score",
        ]
    ]
    .sort_values(
        "divergence_score",
        ascending=False
    )
    .head(50)
)

print(
    interesting.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

interesting.to_csv(
    TABLES_DIR
    / "vader_transformer_high_divergence.csv",
    index=False
)


# ============================================================
# PART H
# EXTREME CROSS-MODEL CASES
# ============================================================

print("\n" + "=" * 70)
print("PART H — CROSS-MODEL CONTRASTS")
print("=" * 70)

contrast = df[
    (
        (
            df["vader_label"]
            == "positive"
        )
        &
        (
            df["emotion_label"]
            .isin(
                [
                    "anger",
                    "fear",
                    "sadness",
                    "disgust"
                ]
            )
        )
    )
    |
    (
        (
            df["vader_label"]
            == "negative"
        )
        &
        (
            df["emotion_label"]
            == "joy"
        )
    )
].copy()

print(
    f"\nPositive VADER + negative-valence emotion "
    f"cases, or negative VADER + joy cases: "
    f"{len(contrast):,}"
)

print(
    contrast[
        [
            "record_id",
            "community",
            "vader_label",
            "vader_compound",
            "emotion_label",
            "emotion_confidence",
        ]
    ]
    .head(50)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

contrast.to_csv(
    TABLES_DIR
    / "vader_transformer_cross_model_contrasts.csv",
    index=False
)


# ============================================================
# PART I
# 2016 ROBUSTNESS
# ============================================================

print("\n" + "=" * 70)
print("PART I — 2016-ONLY AGREEMENT")
print("=" * 70)

df["year"] = pd.to_datetime(
    df["created_utc"]
).dt.year

df2016 = df[
    df["year"] == 2016
].copy()

v_neutral_2016 = (
    df2016["vader_label"]
    == "neutral"
)

t_neutral_2016 = (
    df2016["emotion_label"]
    == "neutral"
)

agreement2016 = (
    v_neutral_2016
    == t_neutral_2016
)

kappa2016 = cohen_kappa_score(
    v_neutral_2016,
    t_neutral_2016
)

print(
    f"\n2016 observations: "
    f"{len(df2016):,}"
)

print(
    f"2016 neutral agreement: "
    f"{agreement2016.mean() * 100:.2f}%"
)

print(
    f"2016 neutral kappa: "
    f"{kappa2016:.4f}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VADER ↔ TRANSFORMER ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    TABLES_DIR
)

print(
    "\nNo source corpus was modified."
)

print(
    "\nNext stage:"
)

print(
    "Linguistic / psychological language features"
)

print("=" * 70)