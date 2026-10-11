import pandas as pd
from pathlib import Path

p = Path("data/processed/linanqiu")

print("=" * 80)
print("PROCESSED DATA QUALITY CHECK")
print("=" * 80)

for f in sorted(p.glob("*.csv")):
    df = pd.read_csv(f)

    print()
    print(f.name)
    print("Rows:", len(df))
    print("Minimum date:", df["created_utc"].min())
    print("Maximum date:", df["created_utc"].max())
    print("Missing text:", df["text"].isna().sum())
    print("Missing time:", df["created_utc"].isna().sum())

print()
print("=" * 80)
