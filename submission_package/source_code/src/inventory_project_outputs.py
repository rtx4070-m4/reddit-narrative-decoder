from pathlib import Path
import pandas as pd

root = Path(".")
outdir = root / "results" / "tables" / "phase11_final_report"
outdir.mkdir(parents=True, exist_ok=True)

rows = []

for base in [root / "results", root / "src"]:
    if not base.exists():
        continue
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        if any(part in {".venv", "__pycache__"} for part in p.parts):
            continue
        rows.append({
            "area": str(base),
            "relative_path": str(p.relative_to(root)),
            "extension": p.suffix.lower(),
            "size_bytes": p.stat().st_size,
        })

df = pd.DataFrame(rows)
target = outdir / "project_file_inventory.csv"
df.to_csv(target, index=False)

print("Files inventoried:", len(df))
print("Inventory saved:", target.resolve())
print("\nFiles by extension:")
print(df.groupby(["area", "extension"]).size().to_string())
print("\nTop-level results directories:")
results = root / "results" / "tables"
if results.exists():
    for p in sorted(results.iterdir()):
        if p.is_dir():
            print(" -", p.name)
