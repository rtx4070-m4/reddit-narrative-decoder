from pathlib import Path
import pandas as pd

path = Path(
    "results/tables/phase10g_topic_interpretability_audit/"
    "phase10g_sampled_topic_examples.csv"
)

df = pd.read_csv(path)

print("=" * 72)
print("PHASE 10G INTERPRETABILITY REVIEW AUDIT")
print("=" * 72)
print("Total sampled examples:", len(df))
print("Columns:", list(df.columns))

for col in ["human_topic_label", "interpretability_rating", "review_notes"]:
    if col in df.columns:
        values = df[col].fillna("").astype(str).str.strip()
        populated = values.ne("")
        print(
            f"{col}: populated={int(populated.sum())}, "
            f"blank={int((~populated).sum())}"
        )
        if populated.any():
            print("Existing values:")
            print(values[populated].value_counts().to_string())

print("\nSamples per model and community:")
print(
    df.groupby(["model", "community"])
      .size()
      .rename("sample_rows")
      .to_string()
)

print("\nSamples per model and topic:")
print(
    df.groupby(["model", "topic"])
      .size()
      .rename("sample_rows")
      .to_string()
)

print("\nAvailable example text by model and topic:")
for (model, topic), group in df.groupby(["model", "topic"], sort=True):
    print(f"\n--- {model} | {topic} | n={len(group)} ---")
    for _, row in group.head(3).iterrows():
        community = row.get("community", "")
        text = str(row.get("review_text", "")).replace("\n", " ")
        print(f"[{community}] {text[:350]}")

