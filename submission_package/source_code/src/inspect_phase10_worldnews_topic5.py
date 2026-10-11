import pandas as pd

path = "results/tables/phase10o_human_validation/phase10o_example_rating_sheet.csv"
df = pd.read_csv(path)

subset = df[
    (df["model"] == "Semantic_KMeans") &
    (df["community"] == "worldnews") &
    (df["topic"] == "Topic 5")
]

print("Semantic_KMeans | worldnews | Topic 5")
print("Suggested label: Europe, the Middle East, and displacement")
print("Examples found:", len(subset))
print("-" * 80)

for i, text in enumerate(subset["review_text"].fillna(""), start=1):
    print(f"{i}. {text[:700]}")
