from pathlib import Path
import pandas as pd

root = Path(".")
folders = [
    root / "results/tables/phase7o_temporal_narratives_v17",
    root / "results/tables/phase7j_nmf",
    root / "results/tables/phase8_cross_community",
]

for folder in folders:
    print("\n" + "=" * 85)
    print("FOLDER:", folder)
    print("=" * 85)

    if not folder.exists():
        print("FOLDER MISSING")
        continue

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        print(f"\n{path.name} | {path.stat().st_size:,} bytes")

        if path.suffix.lower() == ".csv" and path.stat().st_size:
            try:
                df = pd.read_csv(path, nrows=3)
                print("Columns:", list(df.columns))
            except Exception as exc:
                print("CSV READ ERROR:", exc)

print("\nInspection complete. No files were modified.")
