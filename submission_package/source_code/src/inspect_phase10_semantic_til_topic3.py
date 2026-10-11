import pandas as pd

path = "results/tables/phase10o_human_validation/phase10o_example_rating_sheet.csv"
df = pd.read_csv(path)

subset = df[
    (df["model"] == "Semantic_KMeans") &
    (df["community"] == "todayilearned") &
    (df["topic"] == "Topic 3")
]

print("Semantic_KMeans | todayilearned | Topic 3")
print("Suggested label: Art and history references")
print("Examples found:", len(subset))
print("-" * 80)

for i, text in enumerate(subset["review_text"].fillna(""), start=1):
    print(f"{i}. {text[:700]}")
