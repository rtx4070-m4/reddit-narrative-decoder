from pathlib import Path
import pandas as pd
import numpy as np

root = Path(".")
base = root / "results/tables/phase7o_temporal_narratives_v17"
out = root / "results/tables/phase12a_statistical_audit"
out.mkdir(parents=True, exist_ok=True)

nmf_path = base / "nmf_document_topic_date_join.csv"
sem_path = base / "semantic_kmeans_document_topic_date_join.csv"

for path in [nmf_path, sem_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

nmf = pd.read_csv(nmf_path)
sem = pd.read_csv(sem_path)

required = ["record_id", "analysis_date", "community", "has_assignment"]
for name, df in [("NMF", nmf), ("Semantic K-means", sem)]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} file missing columns: {missing}")

def assignment_bool(series):
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes"]
    )

frames = []
for model, df in [("NMF", nmf), ("Semantic K-means", sem)]:
    d = df.copy()
    d["model"] = model
    d["date"] = pd.to_datetime(d["analysis_date"], errors="coerce")
    d["month"] = d["date"].dt.to_period("M").astype("string")
    d["assigned"] = assignment_bool(d["has_assignment"])
    frames.append(d)

all_data = pd.concat(frames, ignore_index=True)

# A. Monthly full-corpus coverage for each model
monthly = (
    all_data.dropna(subset=["month"])
    .groupby(["model", "month"], as_index=False)
    .agg(
        corpus_rows=("record_id", "size"),
        unique_record_ids=("record_id", "nunique"),
        assigned_rows=("assigned", "sum"),
        dated_rows=("date", "count"),
    )
)
monthly["unassigned_rows"] = monthly["corpus_rows"] - monthly["assigned_rows"]
monthly["assignment_coverage"] = (
    monthly["assigned_rows"] / monthly["corpus_rows"]
)
monthly["assigned_share_of_corpus"] = (
    monthly["assigned_rows"] / monthly["corpus_rows"]
)
monthly.to_csv(out / "phase12d_monthly_coverage.csv", index=False)

# B. Monthly counts by community and model
community_monthly = (
    all_data.dropna(subset=["month", "community"])
    .groupby(["model", "community", "month"], as_index=False)
    .agg(
        corpus_rows=("record_id", "size"),
        assigned_rows=("assigned", "sum"),
        unique_record_ids=("record_id", "nunique"),
    )
)
community_monthly["unassigned_rows"] = (
    community_monthly["corpus_rows"] - community_monthly["assigned_rows"]
)
community_monthly["assignment_coverage"] = (
    community_monthly["assigned_rows"] / community_monthly["corpus_rows"]
)
community_monthly.to_csv(
    out / "phase12d_community_monthly_coverage.csv", index=False
)

# C. Timeline gaps and month completeness
valid_dates = all_data.dropna(subset=["date"])
if valid_dates.empty:
    raise ValueError("No valid dates found.")

start = valid_dates["date"].min().to_period("M")
end = valid_dates["date"].max().to_period("M")
expected_months = pd.period_range(start, end, freq="M")

observed = sorted(
    valid_dates["date"].dt.to_period("M").unique()
)
observed_set = set(observed)
missing_months = [str(m) for m in expected_months if m not in observed_set]

timeline = pd.DataFrame({"month": [str(m) for m in expected_months]})
for model, df in [("NMF", nmf), ("Semantic K-means", sem)]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["analysis_date"], errors="coerce")
    d["month"] = d["date"].dt.to_period("M").astype("string")
    counts = d.groupby("month").size()
    column_name = "nmf_rows" if model == "NMF" else "semantic_kmeans_rows"
    timeline[column_name] = (
        timeline["month"].map(counts).fillna(0).astype(int)
    )
timeline["month_observed_in_both_models"] = (
    (timeline["nmf_rows"] > 0) &
    (timeline["semantic_kmeans_rows"] > 0)
)
timeline.to_csv(out / "phase12d_timeline_completeness.csv", index=False)

# D. Summary metrics
summary = []
for model, df in [("NMF", nmf), ("Semantic K-means", sem)]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["analysis_date"], errors="coerce")
    d["month"] = d["date"].dt.to_period("M").astype("string")
    d["assigned"] = assignment_bool(d["has_assignment"])
    dated = d.dropna(subset=["month"])
    counts = dated.groupby("month").size()
    valid = dated["date"].notna()

    summary.append({
        "model": model,
        "total_rows": len(d),
        "unique_record_ids": d["record_id"].nunique(),
        "duplicate_record_id_rows": int(d["record_id"].duplicated().sum()),
        "missing_dates": int(d["date"].isna().sum()),
        "first_date": d["date"].min(),
        "last_date": d["date"].max(),
        "months_with_records": int(counts.size),
        "months_in_full_timeline": len(expected_months),
        "months_with_zero_records": len(missing_months),
        "minimum_monthly_rows": int(counts.min()) if len(counts) else 0,
        "median_monthly_rows": float(counts.median()) if len(counts) else 0,
        "maximum_monthly_rows": int(counts.max()) if len(counts) else 0,
        "rows_in_largest_month": int(counts.max()) if len(counts) else 0,
        "assignment_coverage_overall": float(d["assigned"].mean()),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(out / "phase12d_temporal_summary.csv", index=False)

# E. Explicit small-month flag; threshold is a diagnostic, not a statistical rule.
monthly["small_month_under_30_rows"] = monthly["corpus_rows"] < 30
monthly.to_csv(out / "phase12d_monthly_coverage.csv", index=False)

report = [
    "# Phase 12D — Temporal Sample Imbalance Audit",
    "",
    "## Summary",
    summary_df.to_string(index=False),
    "",
    "## Months with fewer than 30 records",
    monthly[monthly["small_month_under_30_rows"]].to_string(index=False),
    "",
    "## Missing months",
    ", ".join(missing_months) if missing_months else "No completely missing months in the observed date range.",
    "",
    "## Interpretation safeguards",
    "- Monthly counts describe the available corpus, not Reddit activity across the entire platform.",
    "- Months with very small samples can produce unstable topic proportions.",
    "- A change in topic share does not by itself establish a change in emotional expression or a causal narrative shift.",
    "- Compare all-corpus shares with shares among assigned documents, and disclose unassigned records.",
    "- The threshold of 30 rows is a flag for review, not a universal adequacy criterion.",
    "- The audit does not change model assignments or original Phase 7O outputs.",
]
(out / "phase12d_temporal_audit_report.md").write_text(
    "\n".join(report), encoding="utf-8"
)

print("\n=== PHASE 12D TEMPORAL SUMMARY ===")
print(summary_df.to_string(index=False))
print("\n=== MONTHLY COVERAGE ===")
print(monthly.to_string(index=False))
print("\n=== TIMELINE COMPLETENESS ===")
print(timeline.to_string(index=False))
print("\nMissing months:", missing_months)
print("\nSaved outputs to:", out.resolve())

