from pathlib import Path
import re
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

base = Path("results/tables/phase8_cross_community")
input_file = base / "phase8_merged_observations.csv"
df = pd.read_csv(input_file)

# Confirm required columns exist.
required = {"community", "vader_label", "emotion_label"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Compare neutral versus non-neutral.
df["vader_binary"] = df["vader_label"].map(
    lambda x: "neutral" if x == "neutral" else "non-neutral"
)
df["emotion_binary"] = df["emotion_label"].map(
    lambda x: "neutral" if x == "neutral" else "non-neutral"
)

if df[["vader_binary", "emotion_binary"]].isna().any().any():
    raise ValueError("Missing binary labels detected.")

labels = ["neutral", "non-neutral"]
summary_rows = []

for community, data in df.groupby("community", sort=True):
    cm = confusion_matrix(
        data["vader_binary"],
        data["emotion_binary"],
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=["VADER_neutral", "VADER_non_neutral"],
        columns=["Transformer_neutral", "Transformer_non_neutral"]
    )

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(community)).strip("_")
    cm_path = base / f"phase8_agreement_confusion_{safe_name}.csv"
    cm_df.to_csv(cm_path)

    agreement = accuracy_score(data["vader_binary"], data["emotion_binary"])
    kappa = cohen_kappa_score(data["vader_binary"], data["emotion_binary"])

    summary_rows.append({
        "community": community,
        "n": len(data),
        "agreement_count": int(
            (data["vader_binary"] == data["emotion_binary"]).sum()
        ),
        "observed_agreement": agreement,
        "observed_agreement_percent": agreement * 100,
        "cohen_kappa": kappa,
        "vader_neutral_n": int((data["vader_binary"] == "neutral").sum()),
        "transformer_neutral_n": int(
            (data["emotion_binary"] == "neutral").sum()
        ),
        "confusion_matrix_file": cm_path.name
    })

summary = pd.DataFrame(summary_rows)
summary_path = base / "phase8_vader_transformer_agreement_by_community.csv"
summary.to_csv(summary_path, index=False)

print("\nAGREEMENT BY COMMUNITY")
print(summary.to_string(index=False))

print("\nCONFUSION MATRICES")
for row in summary_rows:
    print(f"\nCommunity: {row['community']}")
    print(pd.read_csv(base / row["confusion_matrix_file"], index_col=0).to_string())

print("\nValidation:")
print("Community sample sizes sum to:", int(summary["n"].sum()))
print("Merged dataset rows:", len(df))
print("Sample-size check:", "PASS" if summary["n"].sum() == len(df) else "FAIL")
print("\nSaved summary:", summary_path)
