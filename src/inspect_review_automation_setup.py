from pathlib import Path
import sys
import importlib.util

root = Path.cwd()

print("=" * 70)
print("REDDIT NARRATIVE DECODER — READ-ONLY INSPECTION")
print("=" * 70)
print("Project:", root)
print("Python:", sys.executable)
print("Python version:", sys.version.split()[0])

targets = [
    "results/tables/phase10o_human_validation/phase10o_example_rating_sheet.csv",
    "results/tables/phase10o_human_validation/phase10o_group_decision_sheet.csv",
    "results/tables/phase10p_review_pack/phase10p_group_evidence_review.csv",
    "results/tables/phase10s_label_revision/phase10s_group_revision_sheet.csv",
    "results/tables/phase10s_label_revision/phase10s_integrity_summary.csv",
]

print("\nREVIEW FILES")
for item in targets:
    path = root / item
    print(f"{'FOUND' if path.is_file() else 'MISSING'}: {item}")
    if path.is_file():
        print(f"  Size: {path.stat().st_size:,} bytes")

print("\nPYTHON PACKAGES")
for package in [
    "pandas",
    "dotenv",
    "openai",
    "anthropic",
    "requests",
]:
    print(f"{package}: {'installed' if importlib.util.find_spec(package) else 'not installed'}")

print("\nEXISTING REVIEW-RUNNER FILES")
src = root / "src"
if src.is_dir():
    matches = sorted(
        p.name for p in src.iterdir()
        if any(term in p.name.lower() for term in [
            "phase10", "review", "decision", "label"
        ])
    )
    for name in matches:
        print(" ", name)
    if not matches:
        print("No matching filenames found.")
else:
    print("src folder not found.")

print("\nInspection complete. No files were modified.")
