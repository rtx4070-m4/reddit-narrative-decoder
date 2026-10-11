from pathlib import Path
import pandas as pd

path = Path("results/tables/phase10o_human_validation/phase10_review_decision_log.csv")

print("Decision log exists:", path.exists())

if path.exists():
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    print("Rows:", len(df))
    print(df[["model", "community", "topic", "decision"]].to_string(index=False))
else:
    print("No saved decisions found. Original review files remain separate.")
