import pandas as pd
from pathlib import Path

p = Path("data/raw/linanqiu")

print("=" * 80)
print("RAW COLUMN MAPPING CHECK")
print("=" * 80)

for f in p.glob("*.csv"):
    print()
    print(f.name)

    df = pd.read_csv(f, nrows=3)

    print(df.iloc[:, :12].to_string(index=False))

print("=" * 80)
