from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

base = Path("results/tables/phase8_cross_community")
df = pd.read_csv(base / "phase8_merged_observations.csv")

# Create comparable binary categories.
df["vader_binary"] = df["vader_label"].map(
    lambda x: "neutral" if x == "neutral" else "non-neutral"
)
df["emotion_binary"] = df["emotion_label"].map(
    lambda x: "neutral" if x == "neutral" else "non-neutral"
)

periods = {
    "february_2016": df[df["is_february_2016"] == True],
    "2016_only": df[df["is_2016"] == True],
    "all_years": df
}

labels = ["neutral", "non-neutral"]
summary_rows = []

for period, data in periods.items():
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

    cm_path = base / f"phase8_vader_transformer_binary_confusion_matrix_{period}.csv"
    cm_df.to_csv(cm_path)

    agreement = accuracy_score(data["vader_binary"], data["emotion_binary"])
    kappa = cohen_kappa_score(data["vader_binary"], data["emotion_binary"])

    summary_rows.append({
        "period": period,
        "n": len(data),
        "agreement_count": int((data["vader_binary"] == data["emotion_binary"]).sum()),
        "observed_agreement": agreement,
        "observed_agreement_percent": agreement * 100,
        "cohen_kappa": kappa,
        "vader_neutral_n": int((data["vader_binary"] == "neutral").sum()),
        "transformer_neutral_n": int((data["emotion_binary"] == "neutral").sum()),
        "confusion_matrix_file": cm_path.name
    })

summary = pd.DataFrame(summary_rows)
summary_path = base / "phase8_vader_transformer_binary_agreement_by_period.csv"
summary.to_csv(summary_path, index=False)

print("\nAGREEMENT COMPARISON")
print(summary.to_string(index=False))

for period in periods:
    path = base / f"phase8_vader_transformer_binary_confusion_matrix_{period}.csv"
    print(f"\nCONFUSION MATRIX — {period}")
    print(pd.read_csv(path, index_col=0).to_string())

print("\nSaved summary:", summary_path)
