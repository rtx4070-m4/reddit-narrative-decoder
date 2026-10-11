from pathlib import Path
import pandas as pd

root = Path(".")
out = root / "results/tables/phase12a_statistical_audit"
out.mkdir(parents=True, exist_ok=True)

sources = [
    {
        "name": "Phase 7J cleaned corpus",
        "path": root / "results/tables/phase7j_cleaned_documents_audit.csv",
        "date_candidates": ["date", "created_utc", "analysis_date"],
        "id_candidates": ["record_id", "id"],
    },
    {
        "name": "Phase 8 merged observations",
        "path": root / "results/tables/phase8_cross_community/phase8_merged_observations.csv",
        "date_candidates": ["created_utc", "date", "analysis_date"],
        "id_candidates": ["record_id", "id"],
    },
    {
        "name": "Phase 7O NMF temporal join",
        "path": root / "results/tables/phase7o_temporal_narratives_v17/nmf_document_topic_date_join.csv",
        "date_candidates": ["analysis_date", "date", "created_utc"],
        "id_candidates": ["record_id", "id"],
    },
    {
        "name": "Phase 7O Semantic temporal join",
        "path": root / "results/tables/phase7o_temporal_narratives_v17/semantic_kmeans_document_topic_date_join.csv",
        "date_candidates": ["analysis_date", "date", "created_utc"],
        "id_candidates": ["record_id", "id"],
    },
]

monthly_frames = []
summary = []

for source in sources:
    name = source["name"]
    path = source["path"]

    if not path.exists():
        summary.append({
            "source": name,
            "status": "MISSING",
            "rows": None,
            "unique_ids": None,
            "date_column": None,
            "missing_dates": None,
            "first_date": None,
            "last_date": None,
            "detail": f"File not found: {path}",
        })
        continue

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        summary.append({
            "source": name,
            "status": "READ_ERROR",
            "rows": None,
            "unique_ids": None,
            "date_column": None,
            "missing_dates": None,
            "first_date": None,
            "last_date": None,
            "detail": repr(exc),
        })
        continue

    date_col = next(
        (c for c in source["date_candidates"] if c in df.columns), None
    )
    id_col = next(
        (c for c in source["id_candidates"] if c in df.columns), None
    )

    if date_col is None:
        summary.append({
            "source": name,
            "status": "REVIEW",
            "rows": len(df),
            "unique_ids": df[id_col].nunique() if id_col else None,
            "date_column": None,
            "missing_dates": None,
            "first_date": None,
            "last_date": None,
            "detail": f"No expected date column. Available columns: {list(df.columns)}",
        })
        continue

    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    months = dates.dt.strftime("%Y-%m")

    summary.append({
        "source": name,
        "status": "READ_OK",
        "rows": len(df),
        "unique_ids": df[id_col].nunique() if id_col else None,
        "date_column": date_col,
        "missing_dates": int(dates.isna().sum()),
        "first_date": str(dates.min()) if dates.notna().any() else None,
        "last_date": str(dates.max()) if dates.notna().any() else None,
        "detail": f"Columns detected; ID column={id_col}",
    })

    valid = pd.DataFrame({"month": months}).dropna()
    counts = valid.groupby("month").size().rename("rows").reset_index()
    counts["source"] = name
    monthly_frames.append(counts)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(out / "phase12e_source_date_summary.csv", index=False)

if monthly_frames:
    monthly = pd.concat(monthly_frames, ignore_index=True)
    pivot = monthly.pivot_table(
        index="month", columns="source", values="rows",
        aggfunc="sum", fill_value=0
    ).reset_index()
    pivot.to_csv(out / "phase12e_source_monthly_comparison.csv", index=False)
else:
    pivot = pd.DataFrame()

report = [
    "# Phase 12E — Source Date and Monthly Distribution Audit",
    "",
    "## Source file checks",
    summary_df.to_string(index=False),
    "",
    "## Monthly counts by source",
    pivot.to_string(index=False) if not pivot.empty else "No monthly tables could be created.",
    "",
    "## Interpretation",
    "- If the February concentration appears in the Phase 7J source, it predates the Phase 8 and Phase 7O joins.",
    "- If the distribution changes between sources, investigate filtering, joins, date parsing, and duplicate handling.",
    "- This comparison alone does not establish why the source corpus is concentrated in a particular month.",
    "- No source files are changed by this audit.",
]
(out / "phase12e_source_date_audit_report.md").write_text(
    "\n".join(report), encoding="utf-8"
)

print("\n=== PHASE 12E SOURCE SUMMARY ===")
print(summary_df.to_string(index=False))
print("\n=== MONTHLY DISTRIBUTION COMPARISON ===")
print(pivot.to_string(index=False))
print("\nSaved outputs to:", out.resolve())
