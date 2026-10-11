
from pathlib import Path
import pandas as pd

# ============================================================
# PHASE 10D CORRECTION
# Recalculate monthly coverage and topic prevalence using
# timestamps from the Phase 8 corpus.
# Existing results are not overwritten.
# ============================================================

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"

CORPUS_FILE = (
    TABLES / "phase8_cross_community"
    / "phase8_merged_observations.csv"
)
MODEL_DIR = TABLES / "phase7o_temporal_narratives_v17"

MODELS = {
    "NMF": MODEL_DIR / "nmf_document_topic_date_join.csv",
    "Semantic_KMeans": (
        MODEL_DIR / "semantic_kmeans_document_topic_date_join.csv"
    ),
}

OUT = TABLES / "phase10d_monthly_prevalence_corrected"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 75)
    print("PHASE 10D CORRECTION: MONTHLY PREVALENCE")
    print("=" * 75)

    corpus = pd.read_csv(CORPUS_FILE, low_memory=False)

    if "record_id" not in corpus or "created_utc" not in corpus:
        raise ValueError(
            "Phase 8 must contain record_id and created_utc. "
            f"Available columns: {corpus.columns.tolist()}"
        )

    corpus["record_id"] = corpus["record_id"].astype(str)

    if corpus["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in Phase 8.")

    # Derive month directly from timestamps.
    timestamps = pd.to_datetime(
        corpus["created_utc"], errors="coerce", utc=True
    )
    corpus["audit_month"] = timestamps.dt.strftime("%Y-%m")

    bad_dates = int(corpus["audit_month"].isna().sum())
    print("Phase 8 records:", len(corpus))
    print("Unparseable timestamps:", bad_dates)
    print("Month counts:")
    print(
        corpus["audit_month"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    if corpus["audit_month"].notna().sum() == 0:
        raise ValueError(
            "No usable months could be derived from created_utc. "
            "Inspect the timestamp values before continuing."
        )

    corpus = corpus[corpus["audit_month"].notna()].copy()
    corpus_ids = set(corpus["record_id"])

    monthly_coverage_results = []
    community_month_coverage_results = []
    monthly_topic_results = []
    community_month_topic_results = []

    # Denominators from the full Phase 8 corpus.
    corpus_month = (
        corpus.groupby("audit_month")
        .size()
        .rename("phase8_corpus_records")
        .reset_index()
    )

    corpus_community_month = (
        corpus.groupby(["community", "audit_month"])
        .size()
        .rename("phase8_corpus_records")
        .reset_index()
    )

    for model_name, path in MODELS.items():
        print("\n" + "-" * 75)
        print("MODEL:", model_name)

        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")

        model = pd.read_csv(path, low_memory=False)

        required = ["record_id", "topic_label"]
        missing = [c for c in required if c not in model.columns]
        if missing:
            raise ValueError(f"{model_name} missing columns: {missing}")

        model["record_id"] = model["record_id"].astype(str)

        if model["record_id"].duplicated().any():
            raise ValueError(
                f"{model_name} contains duplicate record IDs."
            )

        unknown_ids = set(model["record_id"]) - corpus_ids
        if unknown_ids:
            raise ValueError(
                f"{model_name} contains {len(unknown_ids)} IDs "
                "not found in Phase 8."
            )

        joined = corpus[
            ["record_id", "community", "audit_month"]
        ].merge(
            model[["record_id", "topic_label"]],
            on="record_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        joined["in_model_file"] = joined["_merge"].eq("both")
        joined["topic_label"] = joined["topic_label"].fillna(
            "OUTSIDE_MODEL_FILE"
        ).astype(str)

        topic_upper = joined["topic_label"].str.strip().str.upper()
        explicit_unassigned = (
            topic_upper.eq("")
            | topic_upper.isin(["NAN", "NONE", "MISSING_TOPIC"])
            | topic_upper.str.contains(
                "UNASSIGNED|UNASSIGN|NO_TOPIC", regex=True, na=False
            )
        )

        joined["assigned"] = (
            joined["in_model_file"] & ~explicit_unassigned
        )

        joined["status"] = "ASSIGNED"
        joined.loc[
            joined["in_model_file"] & ~joined["assigned"], "status"
        ] = "UNASSIGNED_WITHIN_MODEL"
        joined.loc[
            ~joined["in_model_file"], "status"
        ] = "OUTSIDE_MODEL_FILE"

        # ----------------------------------------------------
        # Monthly coverage
        # ----------------------------------------------------
        monthly_status = (
            joined.groupby("audit_month")
            .agg(
                records_in_model_file=("in_model_file", "sum"),
                records_assigned=("assigned", "sum"),
            )
            .reset_index()
        )

        monthly_status = corpus_month.merge(
            monthly_status, on="audit_month", how="left"
        )

        monthly_status[
            ["records_in_model_file", "records_assigned"]
        ] = monthly_status[
            ["records_in_model_file", "records_assigned"]
        ].fillna(0).astype(int)

        monthly_status["model"] = model_name
        monthly_status["records_outside_model_file"] = (
            monthly_status["phase8_corpus_records"]
            - monthly_status["records_in_model_file"]
        )
        monthly_status["unassigned_within_model_file"] = (
            monthly_status["records_in_model_file"]
            - monthly_status["records_assigned"]
        )
        monthly_status["corpus_coverage_pct"] = (
            100 * monthly_status["records_in_model_file"]
            / monthly_status["phase8_corpus_records"]
        )
        monthly_status["assignment_rate_within_model_pct"] = (
            100 * monthly_status["records_assigned"]
            / monthly_status["records_in_model_file"].replace(0, pd.NA)
        )
        monthly_status["assigned_as_pct_of_corpus"] = (
            100 * monthly_status["records_assigned"]
            / monthly_status["phase8_corpus_records"]
        )

        monthly_coverage_results.append(monthly_status)

        # ----------------------------------------------------
        # Community x month coverage
        # ----------------------------------------------------
        cm_status = (
            joined.groupby(["community", "audit_month"])
            .agg(
                records_in_model_file=("in_model_file", "sum"),
                records_assigned=("assigned", "sum"),
            )
            .reset_index()
        )

        cm_status = corpus_community_month.merge(
            cm_status,
            on=["community", "audit_month"],
            how="left",
        )

        cm_status[
            ["records_in_model_file", "records_assigned"]
        ] = cm_status[
            ["records_in_model_file", "records_assigned"]
        ].fillna(0).astype(int)

        cm_status["model"] = model_name
        cm_status["records_outside_model_file"] = (
            cm_status["phase8_corpus_records"]
            - cm_status["records_in_model_file"]
        )
        cm_status["unassigned_within_model_file"] = (
            cm_status["records_in_model_file"]
            - cm_status["records_assigned"]
        )
        cm_status["corpus_coverage_pct"] = (
            100 * cm_status["records_in_model_file"]
            / cm_status["phase8_corpus_records"]
        )
        cm_status["assignment_rate_within_model_pct"] = (
            100 * cm_status["records_assigned"]
            / cm_status["records_in_model_file"].replace(0, pd.NA)
        )
        cm_status["assigned_as_pct_of_corpus"] = (
            100 * cm_status["records_assigned"]
            / cm_status["phase8_corpus_records"]
        )

        community_month_coverage_results.append(cm_status)

        # ----------------------------------------------------
        # Topic prevalence by month among assigned records
        # ----------------------------------------------------
        assigned = joined[joined["assigned"]].copy()

        monthly_topics = (
            assigned.groupby(["audit_month", "topic_label"])
            .size()
            .rename("topic_records")
            .reset_index()
            .rename(columns={"topic_label": "topic"})
        )

        monthly_denoms = monthly_status[
            [
                "audit_month",
                "phase8_corpus_records",
                "records_in_model_file",
                "records_assigned",
            ]
        ]

        monthly_topics = monthly_topics.merge(
            monthly_denoms, on="audit_month", how="left"
        )
        monthly_topics["model"] = model_name

        monthly_topics["pct_among_assigned"] = (
            100 * monthly_topics["topic_records"]
            / monthly_topics["records_assigned"].replace(0, pd.NA)
        )
        monthly_topics["pct_of_model_file"] = (
            100 * monthly_topics["topic_records"]
            / monthly_topics["records_in_model_file"].replace(0, pd.NA)
        )
        monthly_topics["pct_of_phase8_corpus"] = (
            100 * monthly_topics["topic_records"]
            / monthly_topics["phase8_corpus_records"].replace(0, pd.NA)
        )

        monthly_topic_results.append(monthly_topics)

        # ----------------------------------------------------
        # Community x month x topic prevalence
        # ----------------------------------------------------
        cm_topics = (
            assigned.groupby(["community", "audit_month", "topic_label"])
            .size()
            .rename("topic_records")
            .reset_index()
            .rename(columns={"topic_label": "topic"})
        )

        cm_denoms = cm_status[
            [
                "community",
                "audit_month",
                "phase8_corpus_records",
                "records_in_model_file",
                "records_assigned",
            ]
        ]

        cm_topics = cm_topics.merge(
            cm_denoms,
            on=["community", "audit_month"],
            how="left",
        )
        cm_topics["model"] = model_name

        cm_topics["pct_among_assigned"] = (
            100 * cm_topics["topic_records"]
            / cm_topics["records_assigned"].replace(0, pd.NA)
        )
        cm_topics["pct_of_model_file"] = (
            100 * cm_topics["topic_records"]
            / cm_topics["records_in_model_file"].replace(0, pd.NA)
        )
        cm_topics["pct_of_phase8_corpus"] = (
            100 * cm_topics["topic_records"]
            / cm_topics["phase8_corpus_records"].replace(0, pd.NA)
        )

        community_month_topic_results.append(cm_topics)

    outputs = [
        (
            pd.concat(monthly_coverage_results, ignore_index=True),
            "phase10d_corrected_monthly_coverage.csv",
        ),
        (
            pd.concat(community_month_coverage_results, ignore_index=True),
            "phase10d_corrected_community_month_coverage.csv",
        ),
        (
            pd.concat(monthly_topic_results, ignore_index=True),
            "phase10d_corrected_monthly_topic_prevalence.csv",
        ),
        (
            pd.concat(community_month_topic_results, ignore_index=True),
            "phase10d_corrected_community_month_topic_prevalence.csv",
        ),
    ]

    for df, filename in outputs:
        path = OUT / filename
        df.to_csv(path, index=False)
        print("\nSaved:", path)
        print("Rows:", len(df))

    print("\nMONTHLY COVERAGE CHECK")
    coverage = outputs[0][0]
    print(
        coverage[
            [
                "model",
                "audit_month",
                "phase8_corpus_records",
                "records_in_model_file",
                "records_assigned",
                "records_outside_model_file",
                "unassigned_within_model_file",
            ]
        ].to_string(index=False)
    )

    print("\nMONTHLY TOPIC PREVALENCE CHECK")
    topic_table = outputs[2][0]
    feb = topic_table[topic_table["audit_month"] == "2016-02"]
    if feb.empty:
        print("No February 2016 topic rows found; inspect timestamp parsing.")
    else:
        print(feb.to_string(index=False))

    print("\n" + "=" * 75)
    print("MONTHLY PREVALENCE CORRECTION COMPLETED")
    print("Output directory:", OUT)
    print("Original results were not overwritten.")
    print("=" * 75)


if __name__ == "__main__":
    main()
