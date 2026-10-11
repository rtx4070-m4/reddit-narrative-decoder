from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

base = Path("results/tables/phase8_cross_community")
df = pd.read_csv(base / "phase8_merged_observations.csv")

df["vader_binary"] = np.where(
    df["vader_label"].eq("neutral"), "neutral", "non-neutral"
)
df["emotion_binary"] = np.where(
    df["emotion_label"].eq("neutral"), "neutral", "non-neutral"
)
df["agree"] = df["vader_binary"].eq(df["emotion_binary"])

groups = sorted(df["community"].dropna().unique())
rows = []

for community_a, community_b in combinations(groups, 2):
    a = df[df["community"] == community_a]
    b = df[df["community"] == community_b]

    a_agree = int(a["agree"].sum())
    a_disagree = len(a) - a_agree
    b_agree = int(b["agree"].sum())
    b_disagree = len(b) - b_agree

    table = [[a_agree, a_disagree], [b_agree, b_disagree]]
    odds_ratio, p_raw = fisher_exact(table, alternative="two-sided")

    rate_a = a_agree / len(a)
    rate_b = b_agree / len(b)

    rows.append({
        "community_a": community_a,
        "community_b": community_b,
        "n_a": len(a),
        "n_b": len(b),
        "agreement_a_percent": rate_a * 100,
        "agreement_b_percent": rate_b * 100,
        "difference_a_minus_b_percentage_points": (rate_a - rate_b) * 100,
        "odds_ratio_agreement": odds_ratio,
        "p_raw": p_raw
    })

result = pd.DataFrame(rows)

# Holm multiple-comparison correction.
p = result["p_raw"].to_numpy()
m = len(p)
order = np.argsort(p)
adjusted = np.empty(m)
running_max = 0.0

for rank, idx in enumerate(order):
    candidate = (m - rank) * p[idx]
    running_max = max(running_max, candidate)
    adjusted[idx] = min(running_max, 1.0)

result["p_holm"] = adjusted
result["significant_holm_0_05"] = result["p_holm"] < 0.05
result = result.sort_values("p_holm")

out = base / "phase8_agreement_pairwise_community_tests.csv"
result.to_csv(out, index=False)

print("\nPAIRWISE COMMUNITY AGREEMENT TESTS")
print(result.round(6).to_string(index=False))
print("\nSignificant pairs after Holm correction:")
sig = result[result["significant_holm_0_05"]]
print(sig.to_string(index=False) if len(sig) else "None")
print("\nSaved:", out)
