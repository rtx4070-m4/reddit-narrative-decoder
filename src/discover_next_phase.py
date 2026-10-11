from pathlib import Path

root = Path.cwd()
src = root / "src"
results = root / "results" / "tables"

print("=" * 75)
print("REDDIT NARRATIVE DECODER — NEXT PHASE DISCOVERY")
print("=" * 75)

print("\n1. Python scripts in src/")
for path in sorted(src.glob("*.py")):
    print(f"  {path.name}")

print("\n2. Latest phase directories in results/tables/")
phase_dirs = sorted(
    [p for p in results.iterdir() if p.is_dir() and p.name.lower().startswith("phase")],
    key=lambda p: p.name.lower()
)
for path in phase_dirs:
    csv_files = list(path.glob("*.csv"))
    print(f"  {path.name}/  ({len(csv_files)} top-level CSV files)")

print("\n3. Scripts referencing Phase 10 decisions")
needles = [
    "phase10_review_decision_log",
    "phase10o_human_validation",
    "human_decision",
    "final_decision",
    "revised_label",
]
found = False

for path in sorted(src.glob("*.py")):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue

    matches = [term for term in needles if term.lower() in text.lower()]
    if matches:
        found = True
        print(f"\n  {path.name}")
        print("    References:", ", ".join(matches))

if not found:
    print("  No matching downstream references found.")

print("\n4. Phase 10 decision log")
log = results / "phase10o_human_validation" / "phase10_review_decision_log.csv"
print("  Exists:", log.exists())
if log.exists():
    import pandas as pd
    df = pd.read_csv(log, dtype=str, keep_default_na=False)
    print("  Rows:", len(df))
    print("  Columns:", ", ".join(df.columns))

print("\nDiscovery only: no project files were modified.")
