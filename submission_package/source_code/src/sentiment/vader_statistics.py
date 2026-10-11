from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_FILE = (
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

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("VADER STATISTICAL ANALYSIS")
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

required = [
    "record_id",
    "community",
    "vader_compound",
    "vader_label",
    "created_utc",
]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

if df["vader_compound"].isna().any():
    raise ValueError(
        "Missing VADER compound scores detected."
    )


# ============================================================
# COMMUNITY ORDER
# ============================================================

community_order = [
    "gaming",
    "science",
    "todayilearned",
    "worldnews",
]

df["community"] = pd.Categorical(
    df["community"],
    categories=community_order,
    ordered=True
)


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

descriptive = (
    df.groupby(
        "community",
        observed=True
    )["vader_compound"]
    .agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
    )
    .reset_index()
)

descriptive["IQR"] = (
    descriptive["q3"]
    - descriptive["q1"]
)

print(
    descriptive.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

descriptive.to_csv(
    TABLES_DIR
    / "vader_descriptive_statistics.csv",
    index=False
)


# ============================================================
# GROUP DATA
# ============================================================

groups = []

for community in community_order:

    values = (
        df.loc[
            df["community"] == community,
            "vader_compound"
        ]
        .dropna()
        .values
    )

    if len(values) > 0:
        groups.append(values)


# ============================================================
# KRUSKAL-WALLIS TEST
# ============================================================

print("\n" + "=" * 70)
print("KRUSKAL-WALLIS TEST")
print("=" * 70)

h_stat, p_value = stats.kruskal(
    *groups
)

print(
    f"\nH statistic: {h_stat:.4f}"
)

print(
    f"p-value: {p_value:.10g}"
)


# ============================================================
# KRUSKAL-WALLIS EFFECT SIZE
# ============================================================

N = len(df)
k = len(groups)

epsilon_squared = (
    h_stat - k + 1
) / (
    N - k
)

epsilon_squared = max(
    0,
    epsilon_squared
)

print(
    f"Epsilon-squared: "
    f"{epsilon_squared:.6f}"
)


if epsilon_squared < 0.01:
    effect_interpretation = "negligible/small"

elif epsilon_squared < 0.08:
    effect_interpretation = "small"

elif epsilon_squared < 0.26:
    effect_interpretation = "moderate"

else:
    effect_interpretation = "large"


print(
    f"Effect interpretation: "
    f"{effect_interpretation}"
)


kruskal_result = pd.DataFrame({
    "test": ["Kruskal-Wallis"],
    "H_statistic": [h_stat],
    "p_value": [p_value],
    "epsilon_squared": [epsilon_squared],
    "effect_interpretation": [
        effect_interpretation
    ],
    "N": [N],
    "groups": [k],
})

kruskal_result.to_csv(
    TABLES_DIR
    / "vader_kruskal_wallis.csv",
    index=False
)


# ============================================================
# DUNN POST-HOC TEST
# ============================================================

print("\n" + "=" * 70)
print("PAIRWISE POST-HOC COMPARISONS")
print("=" * 70)

try:

    import scikit_posthocs as sp

except ImportError:

    print(
        "\nscikit-posthocs is not installed."
    )

    print(
        "Install it with:"
    )

    print(
        "pip install scikit-posthocs"
    )

    sys.exit(1)


dunn = sp.posthoc_dunn(
    df,
    val_col="vader_compound",
    group_col="community",
    p_adjust="holm"
)

dunn = dunn.reindex(
    index=community_order,
    columns=community_order
)

print(
    "\nDunn test p-values "
    "(Holm adjusted):"
)

print(
    dunn.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)

dunn.to_csv(
    TABLES_DIR
    / "vader_dunn_holm_pvalues.csv"
)


# ============================================================
# PAIRWISE EFFECT SIZE
# ============================================================

print("\n" + "=" * 70)
print("PAIRWISE RANK-BISERIAL EFFECT SIZES")
print("=" * 70)


def rank_biserial(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    n1 = len(x)
    n2 = len(y)

    combined = np.concatenate(
        [x, y]
    )

    ranks = stats.rankdata(
        combined
    )

    r1 = ranks[:n1]

    U1 = (
        np.sum(r1)
        - n1 * (n1 + 1) / 2
    )

    U2 = n1 * n2 - U1

    U = min(U1, U2)

    return 1 - (
        2 * U / (n1 * n2)
    )


pairwise_results = []

for i, group_a in enumerate(
    community_order
):

    for group_b in community_order[
        i + 1:
    ]:

        x = df.loc[
            df["community"] == group_a,
            "vader_compound"
        ].dropna().values

        y = df.loc[
            df["community"] == group_b,
            "vader_compound"
        ].dropna().values

        u_stat, raw_p = stats.mannwhitneyu(
            x,
            y,
            alternative="two-sided"
        )

        effect = rank_biserial(
            x,
            y
        )

        pairwise_results.append({
            "community_1": group_a,
            "community_2": group_b,
            "n_1": len(x),
            "n_2": len(y),
            "mann_whitney_U": u_stat,
            "raw_p": raw_p,
            "rank_biserial": effect,
        })


pairwise_df = pd.DataFrame(
    pairwise_results
)


# ============================================================
# HOLM CORRECTION FOR MANN-WHITNEY TESTS
# ============================================================

from statsmodels.stats.multitest import multipletests

reject, adjusted_p, _, _ = (
    multipletests(
        pairwise_df["raw_p"],
        method="holm"
    )
)

pairwise_df["holm_p"] = adjusted_p
pairwise_df["significant"] = reject


print(
    pairwise_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

pairwise_df.to_csv(
    TABLES_DIR
    / "vader_pairwise_effect_sizes.csv",
    index=False
)


# ============================================================
# 2016 SUBSET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("2016-ONLY ANALYSIS")
print("=" * 70)

df["year"] = (
    df["created_utc"]
    .dt.year
)

df2016 = df[
    df["year"] == 2016
].copy()

print(
    f"\n2016 observations: "
    f"{len(df2016):,}"
)

groups2016 = []

for community in community_order:

    values = (
        df2016.loc[
            df2016["community"] == community,
            "vader_compound"
        ]
        .dropna()
        .values
    )

    if len(values) > 0:
        groups2016.append(values)

h2016, p2016 = stats.kruskal(
    *groups2016
)

N2016 = len(df2016)
k2016 = len(groups2016)

epsilon2016 = max(
    0,
    (h2016 - k2016 + 1)
    / (N2016 - k2016)
)

print(
    f"Kruskal-Wallis H: "
    f"{h2016:.4f}"
)

print(
    f"p-value: "
    f"{p2016:.10g}"
)

print(
    f"Epsilon-squared: "
    f"{epsilon2016:.6f}"
)

dunn2016 = sp.posthoc_dunn(
    df2016,
    val_col="vader_compound",
    group_col="community",
    p_adjust="holm"
)

dunn2016 = dunn2016.reindex(
    index=community_order,
    columns=community_order
)

print(
    "\n2016 Dunn test "
    "(Holm adjusted):"
)

print(
    dunn2016.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)

dunn2016.to_csv(
    TABLES_DIR
    / "vader_dunn_2016_holm_pvalues.csv"
)

pd.DataFrame({
    "analysis": ["all_years", "2016_only"],
    "N": [N, N2016],
    "H_statistic": [
        h_stat,
        h2016
    ],
    "p_value": [
        p_value,
        p2016
    ],
    "epsilon_squared": [
        epsilon_squared,
        epsilon2016
    ],
}).to_csv(
    TABLES_DIR
    / "vader_community_tests.csv",
    index=False
)


# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS COMPLETE")
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
    "\nNext stage: Transformer emotion analysis."
)

print("=" * 70)