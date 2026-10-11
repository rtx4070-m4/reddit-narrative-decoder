from pathlib import Path
import pandas as pd

source = Path(
    "results/tables/phase10g_topic_interpretability_audit/"
    "phase10g_sampled_topic_examples.csv"
)

output = Path(
    "results/tables/phase10g_topic_interpretability_audit/"
    "phase10g_human_review_template.csv"
)

df = pd.read_csv(source)

df.insert(0, "review_row_id", [f"REV_{i:04d}" for i in range(1, len(df) + 1)])

df["reviewer_topic_label"] = ""
df["reviewer_interpretability_rating"] = ""
df["reviewer_notes"] = ""

df["reviewer_interpretability_rating"] = df[
    "reviewer_interpretability_rating"
].astype("string")

output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False, encoding="utf-8-sig")

print("Review template created:", output)
print("Rows:", len(df))
print("Original file unchanged:", source.exists())
print("New columns:", [
    "review_row_id",
    "reviewer_topic_label",
    "reviewer_interpretability_rating",
    "reviewer_notes",
])
print("\nSuggested rating values: Clear; Partially clear; Unclear")
