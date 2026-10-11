import pandas as pd
from pathlib import Path

path = Path("results/tables/phase10o_human_validation/phase10_example_rating_sheet.csv")
if not path.exists():
    path = Path("results/tables/phase10o_human_validation/phase10o_example_rating_sheet.csv")

df = pd.read_csv(path)

groups = [
    ("Semantic KMeans", "science", "Topic 1"),
    ("NMF", "gaming", "Topic 3"),
]

for model, community, topic in groups:
    subset = df[
        (df["model"] == model) &
        (df["community"] == community) &
        (df["topic"] == topic)
    ]
    print("\n" + "=" * 90)
    print(f"{model} | {community} | {topic}")
    print(f"Examples found: {len(subset)}")
    for i, text in enumerate(subset["review_text"].fillna(""), start=1):
        print(f"{i}. {text[:700]}")
