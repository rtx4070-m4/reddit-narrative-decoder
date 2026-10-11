import pandas as pd

path = "results/tables/phase10o_human_validation/phase10o_example_rating_sheet.csv"
df = pd.read_csv(path)

subset = df[
    (df["model"] == "Semantic_KMeans") &
    (df["community"] == "science") &
    (df["topic"] == "Topic 1")
]

print("=" * 90)
print("Semantic_KMeans | science | Topic 1")
print("Suggested label: Automation, employment, and related discussion")
print("Examples found:", len(subset))
print("=" * 90)

for i, text in enumerate(subset["review_text"].fillna(""), start=1):
    print(f"{i}. {text[:700]}")
