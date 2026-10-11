from pathlib import Path
import pandas as pd

root = Path.cwd()
base = root / "results" / "tables"

targets = [
    base / "phase10i_topic_dictionary",
    base / "phase7c_nmf",
    base / "phase7l_semantic",
    base / "phase10o_human_validation",
]

for folder in targets:
    print("\n" + "=" * 75)
    print(folder.relative_to(root))
    print("=" * 75)

    if not folder.exists():
        print("Folder not found")
        continue

    files = sorted(folder.glob("*.csv"))
    if not files:
        print("No top-level CSV files")
        continue

    for path in files:
        print(f"\nFILE: {path.name}")
        try:
            df = pd.read_csv(path, nrows=5)
            print("Columns:", list(df.columns))
            print("Sample rows:")
            print(df.head(3).to_string(index=False, max_cols=12))
        except Exception as exc:
            print("Could not inspect:", exc)

print("\nInspection only. No files were modified.")
