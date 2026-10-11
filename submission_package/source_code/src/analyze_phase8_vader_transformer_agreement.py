from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    classification_report
)

base = Path("results/tables/phase8_cross_community")
input_file = base / "phase8_merged_observations.csv"

df = pd.read_csv(input_file)

# Convert VADER labels to neutral vs non-neutral.
df["vader_binary"] = df["vader_label"].map(
    lambda x: "neutral" if x == "neutral" else "non-neutral"
)

# Convert transformer emotions to neutral vs non-neutral.
df["emotion_binary"] = df["emotion_label"].map(
    lambda x: "neutral" if x == "neutral" else "non-neutral"
)

# Explicit order: rows = VADER, columns = transformer.
labels = ["neutral", "non-neutral"]
cm = confusion_matrix(
    df["vader_binary"],
    df["emotion_binary"],
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=["VADER_neutral", "VADER_non_neutral"],
    columns=["Transformer_neutral", "Transformer_non_neutral"]
)

accuracy = accuracy_score(df["vader_binary"], df["emotion_binary"])
kappa = cohen_kappa_score(df["vader_binary"], df["emotion_binary"])

summary = pd.DataFrame([{
    "period": "all_years",
    "n": len(df),
    "observed_agreement": accuracy,
    "observed_agreement_percent": accuracy * 100,
    "cohen_kappa": kappa,
    "vader_neutral_n": int((df["vader_binary"] == "neutral").sum()),
    "transformer_neutral_n": int((df["emotion_binary"] == "neutral").sum())
}])

cm_path = base / "phase8_vader_transformer_binary_confusion_matrix.csv"
summary_path = base / "phase8_vader_transformer_binary_agreement.csv"

cm_df.to_csv(cm_path)
summary.to_csv(summary_path, index=False)

print("\nCONFUSION MATRIX")
print("Rows = VADER; columns = transformer")
print(cm_df.to_string())

print("\nAGREEMENT SUMMARY")
print(summary.to_string(index=False))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    df["vader_binary"],
    df["emotion_binary"],
    labels=labels,
    zero_division=0
))

print("\nSaved files:")
print(cm_path)
print(summary_path)
