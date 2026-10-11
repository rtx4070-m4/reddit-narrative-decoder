
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PHASE 10D
# DENOMINATOR-CORRECTED TOPIC PREVALENCE AUDIT
#
# Inputs:
#   Phase 8 merged corpus
#   Phase 7O NMF assignment file
#   Phase 7O semantic K-means assignment file
#
# All source files are read-only.
# ============================================================

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"

PHASE8_FILE = (
    TABLES / "phase8_cross_community"
    / "phase8_merged_observations.csv"
)

PHASE7O_DIR = TABLES / "phase7o_temporal_narratives_v17"

MODEL_FILES = {
    "NMF": PHASE7O_DIR / "nmf_document_topic_date_join.csv",
    "Semantic_KMeans": (
        PHASE7O_DIR / "semantic_kmeans_document_topic_date_join.csv"
    ),
}

OUT = TABLES / "phase10d_denominator_corrected_prevalence"
OUT.mkdir(parents=True, exist_ok=True)


def require_columns(df, columns, filename):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def standardize_month(df):
    """Create YYYY-MM month labels from available date fields."""
    df = df.copy()

    if "year" in df.columns and "month" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce")
        month = pd.to_numeric(df["month"], errors="coerce")

        valid = (
            year.notna()
            & month.notna()
            & month.between(1, 12)
        )

        df["audit_month"] = pd.NA
        df.loc[valid, "audit_month"] = (
            year[valid].astype(int).astype(str)
            + "-"
            + month[valid].astype(int).astype(str).str.zfill(2)
        )
        return df

    for col in ["analysis_date", "date", "created_utc"]:
        if col in df.columns:
            dates = pd.to_datetime(
                df[col], errors="coerce", utc=True
            )
            df["audit_month"] = dates.dt.strftime("%Y-%m")
            return df

    raise ValueError(
        "Cannot derive a month. Expected year/month, analysis_date, "
        "date, or created_utc."
    )


def is_unassigned_topic(series):
    """Identify common explicit unassigned labels."""
    labels = series.fillna("").astype(str).str.strip().str.upper()
    return (
        labels.eq("")
        | labels.isin(["NAN", "NONE", "MISSING_TOPIC"])
        | labels.str.contains("UNASSIGNED|UNASSIGN|NO_TOPIC", regex=True)
    )


def safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(len(numerator), dtype=float),
        where=np.asarray(denominator) != 0,
    )


def report_save(df, filename, heading, display_rows=25):
    path = OUT / filename
    df.to_csv(path, index=False)

    print("\n" + "-" * 78)
    print(heading)
    print("Rows:", len(df))
    if len(df):
        print(df.head(display_rows).to_string(index=False))
    print("Saved:", path)


def main():
    print("=" * 78)
    print("PHASE 10D: DENOMINATOR-CORRECTED TOPIC PREVALENCE")
    print("=" * 78)
    print("Project root:", ROOT)

    if not PHASE8_FILE.exists():
        raise FileNotFoundError(f"Phase 8 corpus not found: {PHASE8_FILE}")

    corpus = pd.read_csv(PHASE8_FILE, low_memory=False)
    require_columns(
        corpus, ["record_id", "community"], str(PHASE8_FILE)
    )

    corpus = corpus.copy()
    corpus["record_id"] = corpus["record_id"].astype(str)

    if corpus["record_id"].duplicated().any():
        raise ValueError(
            "Phase 8 has duplicate record IDs. Resolve the ID issue "
            "before calculating prevalence."
        )

    corpus = standardize_month(corpus)

    corpus_ids = set(corpus["record_id"])
    corpus_total = len(corpus)

    print("\nPHASE 8 CORPUS")
    print("Rows:", corpus_total)
    print("Unique record IDs:", len(corpus_ids))
    print("Communities:", sorted(corpus["community"].astype(str).unique()))
    print(
        "Months:",
        sorted(corpus["audit_month"].dropna().astype(str).unique())
    )

    summary_rows = []
    community_coverage_parts = []
    monthly_coverage_parts = []
    topic_overall_parts = []
    topic_community_parts = []
    topic_monthly_parts = []

    # Base denominators from the complete Phase 8 corpus.
    community_base = (
        corpus.groupby("community", dropna=False)
        .size().rename("phase8_corpus_records").reset_index()
    )

    month_base = (
        corpus.groupby("audit_month", dropna=False)
        .size().rename("phase8_corpus_records").reset_index()
    )

    community_month_base = (
        corpus.groupby(["community", "audit_month"], dropna=False)
        .size().rename("phase8_corpus_records").reset_index()
    )

    for model_name, model_path in MODEL_FILES.items():
        print("\n" + "=" * 78)
        print("MODEL:", model_name)
        print("File:", model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected assignment file not found: {model_path}"
            )

        assignments = pd.read_csv(model_path, low_memory=False)
        require_columns(
            assignments,
            ["record_id", "topic_label"],
            str(model_path)
        )

        assignments = assignments.copy()
        assignments["record_id"] = assignments["record_id"].astype(str)

        if assignments["record_id"].duplicated().any():
            raise ValueError(
                f"{model_name} has duplicate record IDs in its assignment file."
            )

        # Verify all model IDs belong to Phase 8.
        model_ids = set(assignments["record_id"])
        unmatched_ids = model_ids - corpus_ids
        if unmatched_ids:
            raise ValueError(
                f"{model_name}: {len(unmatched_ids)} assignment IDs "
                "are absent from Phase 8. Stop and inspect the join."
            )

        # Merge onto the Phase 8 master so denominators and grouping
        # use consistent community/month definitions.
        joined = corpus[
            ["record_id", "community", "audit_month"]
        ].merge(
            assignments[["record_id", "topic_label"]
                        + (["has_assignment"] if "has_assignment" in assignments.columns else [])],
            on="record_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        joined["in_model_file"] = joined["_merge"].eq("both")
        joined["topic_label"] = joined["topic_label"].fillna(
            "OUTSIDE_MODEL_FILE"
        ).astype(str)

        joined["explicit_unassigned"] = is_unassigned_topic(
            joined["topic_label"]
        )

        if "has_assignment" in joined.columns:
            has_assignment = (
                joined["has_assignment"]
                .astype(str).str.strip().str.lower()
                .isin(["true", "1", "yes"])
            )
        else:
            has_assignment = ~joined["explicit_unassigned"]

        joined["has_topic_assignment"] = (
            joined["in_model_file"]
            & ~joined["explicit_unassigned"]
            & has_assignment
        )

        joined["assignment_status"] = np.select(
            [
                ~joined["in_model_file"],
                joined["in_model_file"]
                & ~joined["has_topic_assignment"],
            ],
            [
                "OUTSIDE_MODEL_FILE",
                "UNASSIGNED_WITHIN_MODEL",
            ],
            default="ASSIGNED",
        )

        n_in_model = int(joined["in_model_file"].sum())
        n_outside = int((~joined["in_model_file"]).sum())
        n_assigned = int(joined["has_topic_assignment"].sum())
        n_unassigned = int(
            (
                joined["in_model_file"]
                & ~joined["has_topic_assignment"]
            ).sum()
        )

        summary_rows.append({
            "model": model_name,
            "phase8_corpus_records": corpus_total,
            "records_in_model_file": n_in_model,
            "records_outside_model_file": n_outside,
            "records_assigned_to_topic": n_assigned,
            "unassigned_within_model_file": n_unassigned,
            "corpus_coverage_pct": (
                100 * n_in_model / corpus_total if corpus_total else 0
            ),
            "assignment_rate_within_model_pct": (
                100 * n_assigned / n_in_model if n_in_model else 0
            ),
            "assigned_records_as_pct_of_corpus": (
                100 * n_assigned / corpus_total if corpus_total else 0
            ),
        })

        # ----------------------------------------------------
        # A. Community-level coverage and denominators
        # ----------------------------------------------------
        community_model = (
            joined.groupby("community", dropna=False)
            .agg(
                records_in_model_file=("in_model_file", "sum"),
                records_assigned=("has_topic_assignment", "sum"),
            )
            .reset_index()
        )

        community_model["records_in_model_file"] = (
            community_model["records_in_model_file"].astype(int)
        )
        community_model["records_assigned"] = (
            community_model["records_assigned"].astype(int)
        )

        community_unassigned = (
            joined.assign(
                unassigned_inside=(
                    joined["in_model_file"]
                    & ~joined["has_topic_assignment"]
                ).astype(int)
            )
            .groupby("community", dropna=False)["unassigned_inside"]
            .sum().reset_index()
        )

        community_table = (
            community_base
            .merge(community_model, on="community", how="left")
            .merge(community_unassigned, on="community", how="left")
        )

        for col in [
            "records_in_model_file", "records_assigned", "unassigned_inside"
        ]:
            community_table[col] = community_table[col].fillna(0).astype(int)

        community_table["records_outside_model_file"] = (
            community_table["phase8_corpus_records"]
            - community_table["records_in_model_file"]
        )
        community_table["corpus_coverage_pct"] = (
            100 * community_table["records_in_model_file"]
            / community_table["phase8_corpus_records"]
        )
        community_table["assignment_rate_within_model_pct"] = (
            100 * safe_divide(
                community_table["records_assigned"],
                community_table["records_in_model_file"],
            )
        )
        community_table["assigned_as_pct_of_corpus"] = (
            100 * community_table["records_assigned"]
            / community_table["phase8_corpus_records"]
        )
        community_table.insert(0, "model", model_name)
        community_coverage_parts.append(community_table)

        # ----------------------------------------------------
        # B. Monthly coverage and denominators
        # ----------------------------------------------------
        monthly_model = (
            joined.groupby("audit_month", dropna=False)
            .agg(
                records_in_model_file=("in_model_file", "sum"),
                records_assigned=("has_topic_assignment", "sum"),
            )
            .reset_index()
        )

        monthly_model["records_in_model_file"] = (
            monthly_model["records_in_model_file"].astype(int)
        )
        monthly_model["records_assigned"] = (
            monthly_model["records_assigned"].astype(int)
        )

        monthly_unassigned = (
            joined.assign(
                unassigned_inside=(
                    joined["in_model_file"]
                    & ~joined["has_topic_assignment"]
                ).astype(int)
            )
            .groupby("audit_month", dropna=False)["unassigned_inside"]
            .sum().reset_index()
        )

        monthly_table = (
            month_base
            .merge(monthly_model, on="audit_month", how="left")
            .merge(monthly_unassigned, on="audit_month", how="left")
        )

        for col in [
            "records_in_model_file", "records_assigned", "unassigned_inside"
        ]:
            monthly_table[col] = monthly_table[col].fillna(0).astype(int)

        monthly_table["records_outside_model_file"] = (
            monthly_table["phase8_corpus_records"]
            - monthly_table["records_in_model_file"]
        )
        monthly_table["corpus_coverage_pct"] = (
            100 * monthly_table["records_in_model_file"]
            / monthly_table["phase8_corpus_records"]
        )
        monthly_table["assignment_rate_within_model_pct"] = (
            100 * safe_divide(
                monthly_table["records_assigned"],
                monthly_table["records_in_model_file"],
            )
        )
        monthly_table["assigned_as_pct_of_corpus"] = (
            100 * monthly_table["records_assigned"]
            / monthly_table["phase8_corpus_records"]
        )
        monthly_table.insert(0, "model", model_name)
        monthly_coverage_parts.append(monthly_table)

        # ----------------------------------------------------
        # C. Topic prevalence, excluding unassigned rows
        # ----------------------------------------------------
        assigned = joined[joined["has_topic_assignment"]].copy()

        overall_counts = (
            assigned.groupby("topic_label", dropna=False)
            .size().rename("topic_records").reset_index()
            .rename(columns={"topic_label": "topic"})
        )

        overall_counts["model"] = model_name
        overall_counts["assigned_records_denominator"] = n_assigned
        overall_counts["model_file_denominator"] = n_in_model
        overall_counts["phase8_corpus_denominator"] = corpus_total

        overall_counts["pct_among_assigned"] = (
            100 * overall_counts["topic_records"] / n_assigned
            if n_assigned else 0
        )
        overall_counts["pct_of_model_file"] = (
            100 * overall_counts["topic_records"] / n_in_model
            if n_in_model else 0
        )
        overall_counts["pct_of_phase8_corpus"] = (
            100 * overall_counts["topic_records"] / corpus_total
            if corpus_total else 0
        )
        topic_overall_parts.append(overall_counts)

        # Topic prevalence within each community.
        comm_topic = (
            assigned.groupby(["community", "topic_label"], dropna=False)
            .size().rename("topic_records").reset_index()
            .rename(columns={"topic_label": "topic"})
        )

        comm_denoms = community_table[
            [
                "community", "phase8_corpus_records",
                "records_in_model_file", "records_assigned"
            ]
        ]

        comm_topic = comm_topic.merge(
            comm_denoms, on="community", how="left"
        )
        comm_topic["model"] = model_name
        comm_topic["pct_among_assigned"] = (
            100 * comm_topic["topic_records"]
            / comm_topic["records_assigned"].replace(0, np.nan)
        )
        comm_topic["pct_of_model_file"] = (
            100 * comm_topic["topic_records"]
            / comm_topic["records_in_model_file"].replace(0, np.nan)
        )
        comm_topic["pct_of_phase8_corpus"] = (
            100 * comm_topic["topic_records"]
            / comm_topic["phase8_corpus_records"].replace(0, np.nan)
        )
        topic_community_parts.append(comm_topic)

        # Topic prevalence within each month.
        month_topic = (
            assigned.groupby(["audit_month", "topic_label"], dropna=False)
            .size().rename("topic_records").reset_index()
            .rename(columns={
                "audit_month": "month",
                "topic_label": "topic"
            })
        )

        month_denoms = monthly_table[
            [
                "audit_month", "phase8_corpus_records",
                "records_in_model_file", "records_assigned"
            ]
        ].rename(columns={"audit_month": "month"})

        month_topic = month_topic.merge(
            month_denoms, on="month", how="left"
        )
        month_topic["model"] = model_name
        month_topic["pct_among_assigned"] = (
            100 * month_topic["topic_records"]
            / month_topic["records_assigned"].replace(0, np.nan)
        )
        month_topic["pct_of_model_file"] = (
            100 * month_topic["topic_records"]
            / month_topic["records_in_model_file"].replace(0, np.nan)
        )
        month_topic["pct_of_phase8_corpus"] = (
            100 * month_topic["topic_records"]
            / month_topic["phase8_corpus_records"].replace(0, np.nan)
        )
        topic_monthly_parts.append(month_topic)

        print("\nModel summary:")
        print("Model-file rows:", n_in_model)
        print("Outside model file:", n_outside)
        print("Assigned topic rows:", n_assigned)
        print("Unassigned within model file:", n_unassigned)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    report_save(
        pd.DataFrame(summary_rows),
        "phase10d_overall_denominator_summary.csv",
        "OVERALL DENOMINATOR SUMMARY",
    )

    report_save(
        pd.concat(community_coverage_parts, ignore_index=True),
        "phase10d_community_coverage.csv",
        "COMMUNITY COVERAGE",
    )

    report_save(
        pd.concat(monthly_coverage_parts, ignore_index=True),
        "phase10d_monthly_coverage.csv",
        "MONTHLY COVERAGE",
        display_rows=40,
    )

    report_save(
        pd.concat(topic_overall_parts, ignore_index=True),
        "phase10d_overall_topic_prevalence.csv",
        "OVERALL TOPIC PREVALENCE",
    )

    report_save(
        pd.concat(topic_community_parts, ignore_index=True),
        "phase10d_community_topic_prevalence.csv",
        "COMMUNITY TOPIC PREVALENCE",
        display_rows=40,
    )

    report_save(
        pd.concat(topic_monthly_parts, ignore_index=True),
        "phase10d_monthly_topic_prevalence.csv",
        "MONTHLY TOPIC PREVALENCE",
        display_rows=40,
    )

    print("\n" + "=" * 78)
    print("PHASE 10D COMPLETED SUCCESSFULLY")
    print("Output directory:", OUT)
    print("Source files were not modified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
