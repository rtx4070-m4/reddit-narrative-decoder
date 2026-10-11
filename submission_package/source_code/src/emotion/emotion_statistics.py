from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


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

COMMUNITIES = [
    "gaming",
    "science",
    "todayilearned",
    "worldnews",
]

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TRANSFORMER EMOTION STATISTICAL ANALYSIS")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["created_utc"]
)

print(
    f"\nObservations: {len(df):,}"
)

print(
    f"Communities: {df['community'].nunique()}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "record_id",
    "community",
    "emotion_label",
    "created_utc",
]

for emotion in EMOTIONS:
    required_columns.append(
        f"emotion_prob_{emotion}"
    )

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# COMMUNITY ORDER
# ============================================================

df["community"] = pd.Categorical(
    df["community"],
    categories=COMMUNITIES,
    ordered=True
)

df["year"] = (
    df["created_utc"]
    .dt.year
)


# ============================================================
# PART A
# CHI-SQUARE TEST OF EMOTION LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PART A — CHI-SQUARE TEST")
print("=" * 70)

contingency = pd.crosstab(
    df["community"],
    df["emotion_label"]
)

contingency = contingency.reindex(
    index=COMMUNITIES,
    columns=EMOTIONS,
    fill_value=0
)

print(
    "\nObserved emotion counts:"
)

print(
    contingency.to_string()
)


chi2, chi_p, chi_df, expected = (
    stats.chi2_contingency(
        contingency
    )
)

print(
    f"\nChi-square statistic: "
    f"{chi2:.4f}"
)

print(
    f"Degrees of freedom: "
    f"{chi_df}"
)

print(
    f"p-value: "
    f"{chi_p:.10g}"
)


# ============================================================
# CRAMER'S V
# ============================================================

n = contingency.to_numpy().sum()

r, c = contingency.shape

cramers_v = np.sqrt(
    chi2
    / (
        n
        * min(r - 1, c - 1)
    )
)

print(
    f"Cramer's V: "
    f"{cramers_v:.6f}"
)


if cramers_v < 0.10:
    cramers_interpretation = "negligible"

elif cramers_v < 0.20:
    cramers_interpretation = "small"

elif cramers_v < 0.40:
    cramers_interpretation = "moderate"

else:
    cramers_interpretation = "large"


print(
    f"Effect interpretation: "
    f"{cramers_interpretation}"
)


chi_square_result = pd.DataFrame({
    "test": ["Chi-square"],
    "chi2": [chi2],
    "df": [chi_df],
    "p_value": [chi_p],
    "cramers_v": [cramers_v],
    "effect_interpretation": [
        cramers_interpretation
    ],
    "N": [n],
})


chi_square_result.to_csv(
    TABLES_DIR
    / "emotion_chi_square.csv",
    index=False
)

contingency.to_csv(
    TABLES_DIR
    / "emotion_contingency_table.csv"
)


# ============================================================
# STANDARDIZED RESIDUALS
# ============================================================

expected_df = pd.DataFrame(
    expected,
    index=COMMUNITIES,
    columns=EMOTIONS
)

residuals = (
    contingency - expected_df
) / np.sqrt(
    expected_df
)

print(
    "\nStandardized residuals:"
)

print(
    residuals.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

residuals.to_csv(
    TABLES_DIR
    / "emotion_standardized_residuals.csv"
)


# ============================================================
# PART B
# KRUSKAL-WALLIS ON EMOTION PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print(
    "PART B — KRUSKAL-WALLIS "
    "FOR EMOTION PROBABILITIES"
)
print("=" * 70)


kw_results = []

for emotion in EMOTIONS:

    column = (
        f"emotion_prob_{emotion}"
    )

    groups = []

    for community in COMMUNITIES:

        values = (
            df.loc[
                df["community"] == community,
                column
            ]
            .dropna()
            .values
        )

        groups.append(values)

    h_stat, p_value = (
        stats.kruskal(
            *groups
        )
    )

    N = len(df)
    k = len(groups)

    epsilon_squared = max(
        0,
        (
            h_stat
            - k
            + 1
        )
        / (
            N
            - k
        )
    )

    kw_results.append({
        "emotion": emotion,
        "H_statistic": h_stat,
        "raw_p": p_value,
        "epsilon_squared": epsilon_squared,
    })


kw_results = pd.DataFrame(
    kw_results
)


# ============================================================
# HOLM CORRECTION
# ============================================================

reject, adjusted_p, _, _ = (
    multipletests(
        kw_results["raw_p"],
        method="holm"
    )
)

kw_results["holm_p"] = (
    adjusted_p
)

kw_results["significant"] = (
    reject
)


print(
    kw_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


kw_results.to_csv(
    TABLES_DIR
    / "emotion_kruskal_wallis.csv",
    index=False
)


# ============================================================
# PART C
# POST-HOC DUNN TESTS FOR SIGNIFICANT EMOTIONS
# ============================================================

print("\n" + "=" * 70)
print("PART C — POST-HOC DUNN TESTS")
print("=" * 70)


try:

    import scikit_posthocs as sp

except ImportError:

    raise ImportError(
        "scikit-posthocs is required. "
        "Run: pip install scikit-posthocs"
    )


significant_emotions = (
    kw_results.loc[
        kw_results["significant"],
        "emotion"
    ]
    .tolist()
)


posthoc_tables = []


for emotion in significant_emotions:

    column = (
        f"emotion_prob_{emotion}"
    )

    temp = df[
        [
            "community",
            column
        ]
    ].dropna()

    dunn = sp.posthoc_dunn(
        temp,
        val_col=column,
        group_col="community",
        p_adjust="holm"
    )

    dunn = dunn.reindex(
        index=COMMUNITIES,
        columns=COMMUNITIES
    )

    output_file = (
        TABLES_DIR
        / f"emotion_dunn_{emotion}.csv"
    )

    dunn.to_csv(
        output_file
    )

    print(
        f"\n{emotion.upper()}"
    )

    print(
        dunn.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )


# ============================================================
# PART D
# 2016-ONLY CHI-SQUARE
# ============================================================

print("\n" + "=" * 70)
print("PART D — 2016-ONLY ANALYSIS")
print("=" * 70)


df2016 = df[
    df["year"] == 2016
].copy()

print(
    f"\n2016 observations: "
    f"{len(df2016):,}"
)


contingency2016 = pd.crosstab(
    df2016["community"],
    df2016["emotion_label"]
)

contingency2016 = (
    contingency2016.reindex(
        index=COMMUNITIES,
        columns=EMOTIONS,
        fill_value=0
    )
)

chi2_2016, p2016, df_2016, expected2016 = (
    stats.chi2_contingency(
        contingency2016
    )
)

n2016 = (
    contingency2016
    .to_numpy()
    .sum()
)

r2016, c2016 = (
    contingency2016.shape
)

cramers_v2016 = np.sqrt(
    chi2_2016
    / (
        n2016
        * min(
            r2016 - 1,
            c2016 - 1
        )
    )
)

print(
    "\n2016 Chi-square:"
)

print(
    f"Chi-square: "
    f"{chi2_2016:.4f}"
)

print(
    f"df: "
    f"{df_2016}"
)

print(
    f"p-value: "
    f"{p2016:.10g}"
)

print(
    f"Cramer's V: "
    f"{cramers_v2016:.6f}"
)


# ============================================================
# 2016 EMOTION PROBABILITY TESTS
# ============================================================

kw2016_results = []

for emotion in EMOTIONS:

    column = (
        f"emotion_prob_{emotion}"
    )

    groups = []

    for community in COMMUNITIES:

        values = (
            df2016.loc[
                df2016["community"] == community,
                column
            ]
            .dropna()
            .values
        )

        groups.append(values)

    h_stat, p_value = (
        stats.kruskal(
            *groups
        )
    )

    kw2016_results.append({
        "emotion": emotion,
        "H_statistic": h_stat,
        "raw_p": p_value,
    })


kw2016_results = pd.DataFrame(
    kw2016_results
)

reject2016, adjusted2016, _, _ = (
    multipletests(
        kw2016_results["raw_p"],
        method="holm"
    )
)

kw2016_results["holm_p"] = (
    adjusted2016
)

kw2016_results["significant"] = (
    reject2016
)

print(
    "\n2016 emotion probability tests:"
)

print(
    kw2016_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

kw2016_results.to_csv(
    TABLES_DIR
    / "emotion_kruskal_wallis_2016.csv",
    index=False
)


# ============================================================
# SAVE 2016 CHI-SQUARE RESULT
# ============================================================

pd.DataFrame({
    "analysis": [
        "all_years",
        "2016_only"
    ],
    "N": [
        n,
        n2016
    ],
    "chi2": [
        chi2,
        chi2_2016
    ],
    "df": [
        chi_df,
        df_2016
    ],
    "p_value": [
        chi_p,
        p2016
    ],
    "cramers_v": [
        cramers_v,
        cramers_v2016
    ],
}).to_csv(
    TABLES_DIR
    / "emotion_community_tests.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("EMOTION STATISTICAL ANALYSIS COMPLETE")
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
    "VADER ↔ Transformer agreement analysis"
)

print("=" * 70)