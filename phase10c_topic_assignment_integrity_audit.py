
from pathlib import Path
import pandas as pd

# ============================================================
# PHASE 10C: TOPIC-ASSIGNMENT INTEGRITY AUDIT
# Does not change source files or topic assignments.
# ============================================================

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"
PHASE8_DIR = TABLES / "phase8_cross_community"
PHASE7O_DIR = TABLES / "phase7o_temporal_narratives_v17"
OUT = TABLES / "phase10c_topic_assignment_audit"
OUT.mkdir(parents=True, exist_ok=True)

PHASE8_FILE = PHASE8_DIR / "phase8_merged_observations.csv"

MODEL_FILES = {
    "NMF": PHASE7O_DIR / "nmf_document_topic_date_join.csv",
    "Semantic_KMeans": (
        PHASE7O_DIR / "semantic_kmeans_document_topic_date_join.csv"
    ),
}


def find_column(df, candidates):
    """Return the first candidate column present in a dataframe."""
    return next((c for c in candidates if c in df.columns), None)


def save_and_print(df, filename, heading):
    path = OUT / filename
    df.to_csv(path, index=False)
    print(f"\n{heading}")
    print(df.to_string(index=False, max_rows=40))
    print("Saved:", path)


def main():
    print("=" * 78)
    print("PHASE 10C: TOPIC-ASSIGNMENT INTEGRITY AUDIT")
    print("=" * 78)

    if not PHASE8_FILE.exists():
        raise FileNotFoundError(f"Phase 8 file not found: {PHASE8_FILE}")

    phase8 = pd.read_csv(PHASE8_FILE, low_memory=False)

    if "record_id" not in phase8.columns:
        raise ValueError("Phase 8 file has no record_id column.")

    print("\nPhase 8 rows:", len(phase8))
    print("Phase 8 unique record IDs:", phase8["record_id"].nunique())

    if phase8["record_id"].duplicated().any():
        print("WARNING: Phase 8 contains repeated record IDs.")

    all_summaries = []
    all_community_tables = []
    all_month_tables = []
    all_join_audits = []

    for model_name, path in MODEL_FILES.items():
        print("\n" + "=" * 78)
        print("MODEL:", model_name)
        print("FILE:", path)

        if not path.exists():
            print("WARNING: File not found; model skipped.")
            continue

        df = pd.read_csv(path, low_memory=False)
        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())

        id_col = find_column(df, ["record_id", "id"])
        topic_col = find_column(
            df, ["topic_label", "topic", "dominant_topic", "cluster"]
        )
        community_col = find_column(df, ["community", "subreddit"])
        month_col = find_column(df, ["month", "year_month", "date", "created_utc"])

        if id_col is None:
            print("WARNING: No record ID column; model skipped.")
            continue

        if topic_col is None:
            print("WARNING: No recognizable topic column; model skipped.")
            continue

        # Standardize audit-only names without modifying source files.
        df = df.copy()
        df["_audit_id"] = df[id_col].astype(str)
        df["_audit_topic"] = (
            df[topic_col].fillna("MISSING_TOPIC").astype(str)
        )

        duplicate_ids = int(df["_audit_id"].duplicated().sum())
        missing_topics = int(
            df["_audit_topic"].isin(
                ["MISSING_TOPIC", "", "nan", "None"]
            ).sum()
        )

        # Identify common UNASSIGNED spellings.
        unassigned_mask = df["_audit_topic"].str.upper().str.contains(
            "UNASSIGNED|UNASSIGN|NO_TOPIC", regex=True, na=False
        )
        unassigned_count = int(unassigned_mask.sum())

        # Compare assignment IDs against the Phase 8 corpus.
        phase8_ids = set(phase8["record_id"].dropna().astype(str))
        model_ids = set(df["_audit_id"].dropna())

        matched_rows = int(df["_audit_id"].isin(phase8_ids).sum())
        unmatched_rows = int((~df["_audit_id"].isin(phase8_ids)).sum())
        unique_model_ids = len(model_ids)

        audit_row = {
            "model": model_name,
            "assignment_rows": len(df),
            "unique_assignment_ids": unique_model_ids,
            "duplicate_id_rows": duplicate_ids,
            "missing_topic_rows": missing_topics,
            "unassigned_rows": unassigned_count,
            "rows_matching_phase8": matched_rows,
            "rows_not_matching_phase8": unmatched_rows,
            "phase8_ids_not_in_model": len(phase8_ids - model_ids),
            "phase8_total_rows": len(phase8),
            "assignment_row_coverage_of_phase8": (
                len(model_ids & phase8_ids) / len(phase8_ids)
                if phase8_ids else 0
            ),
        }
        all_join_audits.append(audit_row)

        # Topic counts: UNASSIGNED remains a category if present.
        topic_counts = (
            df["_audit_topic"].value_counts(dropna=False)
            .rename_axis("topic")
            .reset_index(name="document_count")
        )
        topic_counts.insert(0, "model", model_name)
        topic_counts["share_of_assignment_rows"] = (
            topic_counts["document_count"] / len(df) if len(df) else 0
        )
        all_summaries.append(topic_counts)

        # Community x topic prevalence, if community is available.
        if community_col:
            community_table = (
                df.groupby([community_col, "_audit_topic"], dropna=False)
                .size()
                .reset_index(name="document_count")
                .rename(columns={
                    community_col: "community",
                    "_audit_topic": "topic"
                })
            )
            community_table.insert(0, "model", model_name)

            community_table["community_assignment_total"] = (
                community_table.groupby("community")["document_count"]
                .transform("sum")
            )
            community_table["share_within_community_assignments"] = (
                community_table["document_count"]
                / community_table["community_assignment_total"]
            )
            all_community_tables.append(community_table)

        # Monthly prevalence: use existing monthly/date field if available.
        if month_col:
            if month_col == "created_utc":
                dates = pd.to_datetime(
                    df[month_col], errors="coerce", utc=True
                )
                df["_audit_month"] = dates.dt.strftime("%Y-%m")
            elif month_col == "date":
                dates = pd.to_datetime(df[month_col], errors="coerce")
                df["_audit_month"] = dates.dt.strftime("%Y-%m")
            else:
                df["_audit_month"] = df[month_col].astype(str)

            month_table = (
                df.groupby(["_audit_month", "_audit_topic"], dropna=False)
                .size()
                .reset_index(name="document_count")
                .rename(columns={
                    "_audit_month": "month",
                    "_audit_topic": "topic"
                })
            )
            month_table.insert(0, "model", model_name)
            month_table["month_assignment_total"] = (
                month_table.groupby("month")["document_count"].transform("sum")
            )
            month_table["share_within_month_assignments"] = (
                month_table["document_count"]
                / month_table["month_assignment_total"]
            )
            all_month_tables.append(month_table)

        print("\nKey checks:")
        for key, value in audit_row.items():
            print(f"{key}: {value}")

    if not all_join_audits:
        raise RuntimeError(
            "No model could be audited. Check file names and column names."
        )

    save_and_print(
        pd.DataFrame(all_join_audits),
        "phase10c_join_and_integrity_summary.csv",
        "JOIN AND INTEGRITY SUMMARY"
    )

    if all_summaries:
        save_and_print(
            pd.concat(all_summaries, ignore_index=True),
            "phase10c_topic_counts.csv",
            "TOPIC COUNTS"
        )

    if all_community_tables:
        save_and_print(
            pd.concat(all_community_tables, ignore_index=True),
            "phase10c_community_topic_prevalence.csv",
            "COMMUNITY-LEVEL TOPIC PREVALENCE"
        )

    if all_month_tables:
        save_and_print(
            pd.concat(all_month_tables, ignore_index=True),
            "phase10c_monthly_topic_prevalence.csv",
            "MONTHLY TOPIC PREVALENCE"
        )

    print("\n" + "=" * 78)
    print("PHASE 10C COMPLETED")
    print("Output directory:", OUT)
    print("No source data or model assignment files were modified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
