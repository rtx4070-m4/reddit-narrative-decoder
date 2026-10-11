from pathlib import Path
from collections import Counter
import pandas as pd

root = Path(".")
results = root / "results"
outdir = results / "tables" / "phase11_final_report"
outdir.mkdir(parents=True, exist_ok=True)

file_rows = []
csv_rows = []

for p in sorted(results.rglob("*")):
    if not p.is_file():
        continue
    if any(x in {".venv", "__pycache__"} for x in p.parts):
        continue

    rel = str(p.relative_to(root))
    row = {
        "relative_path": rel,
        "folder": str(p.parent.relative_to(root)),
        "filename": p.name,
        "extension": p.suffix.lower(),
        "size_bytes": p.stat().st_size,
        "empty_file": p.stat().st_size == 0,
    }
    file_rows.append(row)

    if p.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(p, nrows=5)
            full_rows = sum(1 for _ in open(p, encoding="utf-8-sig", errors="replace")) - 1
            csv_rows.append({
                "relative_path": rel,
                "rows_approx": max(full_rows, 0),
                "columns": len(df.columns),
                "column_names": " | ".join(map(str, df.columns)),
                "read_status": "OK",
                "error": "",
            })
        except Exception as exc:
            csv_rows.append({
                "relative_path": rel,
                "rows_approx": None,
                "columns": None,
                "column_names": "",
                "read_status": "ERROR",
                "error": str(exc),
            })

files = pd.DataFrame(file_rows)
csvs = pd.DataFrame(csv_rows)

files.to_csv(outdir / "completion_file_inventory.csv", index=False)
csvs.to_csv(outdir / "completion_csv_audit.csv", index=False)

counts = Counter(files["filename"]) if not files.empty else Counter()
duplicates = [
    {"filename": name, "count": count}
    for name, count in sorted(counts.items()) if count > 1
]
pd.DataFrame(duplicates, columns=["filename", "count"]).to_csv(
    outdir / "completion_duplicate_filenames.csv", index=False
)

phase_folders = sorted({
    str(Path(x).parent) for x in files["relative_path"]
    if str(Path(x).parent).startswith("results")
})
empty_count = int(files["empty_file"].sum()) if not files.empty else 0
error_count = int((csvs["read_status"] == "ERROR").sum()) if not csvs.empty else 0

lines = [
    "# Project Completion Audit",
    "",
    "Read-only structural audit of existing files. File presence does not establish analytical validity.",
    "",
    f"- Files under results: {len(files)}",
    f"- CSV files inspected: {len(csvs)}",
    f"- Empty files: {empty_count}",
    f"- CSV read errors: {error_count}",
    f"- Filenames occurring more than once: {len(duplicates)}",
    "",
    "## Result folders",
]
lines.extend(f"- {x}" for x in phase_folders)
lines += [
    "",
    "## Interpretation cautions",
    "- Duplicate filenames can be legitimate when stored in different folders; inspect paths before treating them as conflicts.",
    "- CSV row counts are approximate physical line counts and may be inaccurate when quoted fields contain newlines.",
    "- Structural checks do not verify statistical assumptions, sample provenance, temporal leakage, or reproducibility.",
    "- Phase 11C assessments are provisional drafts, not completed independent human validation.",
    "- Preserve all existing source data and decisions until discrepancies are investigated.",
    "",
    "## Files generated",
    "- completion_file_inventory.csv",
    "- completion_csv_audit.csv",
    "- completion_duplicate_filenames.csv",
]
(outdir / "completion_audit_report.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)

print("\n".join(lines))
print("\nAudit files saved to:", outdir.resolve())
