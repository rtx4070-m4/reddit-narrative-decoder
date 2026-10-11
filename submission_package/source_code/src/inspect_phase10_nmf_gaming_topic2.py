import pandas as pd

path = "results/tables/phase10o_human_validation/phase10o_example_rating_sheet.csv"
df = pd.read_csv(path)

subset = df[
    (df["model"] == "NMF") &
    (df["community"] == "gaming") &
    (df["topic"] == "Topic 2")
]

print("NMF | gaming | Topic 2")
print("Suggested label: Mixed gaming references and commentary")
print("Examples found:", len(subset))
print("-" * 80)

for i, text in enumerate(subset["review_text"].fillna(""), start=1):
    print(f"{i}. {text[:700]}")
