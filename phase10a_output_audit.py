
from pathlib import Path
import pandas as pd

ROOT = Path(".")
SEARCH_DIRS = [
    ROOT / "results/tables/phase7o_temporal_narratives_v17",
    ROOT / "results/tables/phase8_cross_community",
]

manifest = []

print("=" * 90)
print("PHASE 10A — NARRATIVE OUTPUT AUDIT")
print("=" * 90)

for folder in SEARCH_DIRS:
    print(f"\n\nDIRECTORY: {folder}")

    if not folder.exists():
        print("DIRECTORY NOT FOUND")
        continue

    files = sorted(folder.glob("*.csv"))

    if not files:
        print("No CSV files found.")
        continue

    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)

            record = {
                "directory": str(folder),
                "filename": path.name,
                "rows": len(df),
                "columns_count": len(df.columns),
                "columns": " | ".join(df.columns.astype(str)),
            }
            manifest.append(record)

            print("\n" + "-" * 75)
            print(f"FILE: {path.name}")
            print(f"Rows: {len(df):,}")
            print(f"Columns ({len(df.columns)}):")
            print(", ".join(df.columns.astype(str)))

            # Show a small sample only for potentially relevant tables.
            name = path.name.lower()
            relevant = any(
                term in name
                for term in [
                    "topic", "narrative", "prevalence",
                    "monthly", "temporal", "audit",
                    "assignment", "manifest"
                ]
            )

            if relevant and len(df):
                print("\nSample rows:")
                print(df.head(3).to_string(index=False, max_cols=12))

        except Exception as exc:
            print(f"ERROR reading {path.name}: {exc}")
            manifest.append({
                "directory": str(folder),
                "filename": path.name,
                "rows": None,
                "columns_count": None,
                "columns": f"ERROR: {exc}",
            })

if manifest:
    output = ROOT / "results/tables/phase10a_output_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(output, index=False)
    print("\n" + "=" * 90)
    print(f"Manifest saved to: {output}")
    print(f"Files inventoried: {len(manifest)}")
