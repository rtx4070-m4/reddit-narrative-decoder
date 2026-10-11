
from pathlib import Path
import pandas as pd

# ============================================================
# PHASE 10E: FEBRUARY 2016 CROSS-COMMUNITY TOPIC COMPARISON
# Uses correct corpus, model-file, and assigned-record denominators.
# Source files are read-only.
# ============================================================

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"

CORPUS_FILE = (
    TABLES / "phase8_cross_community"
    / "phase8_merged_observations.csv"
)
MODEL_DIR = TABLES / "phase7o_temporal_narratives_v17"
OUT = TABLES / "phase10e_february_cross_community"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "NMF": MODEL_DIR / "nmf_document_topic_date_join.csv",
    "Semantic_KMeans": (
        MODEL_DIR / "semantic_kmeans_document_topic_date_join.csv"
    ),
}


def main():
    print("=" * 75)
    print("PHASE 10E: FEBRUARY CROSS-COMMUNITY TOPIC COMPARISON")
    print("=" * 75)

    corpus = pd.read_csv(CORPUS_FILE, low_memory=False)

    required = ["record_id", "community", "created_utc"]
    missing = [c for c in required if c not in corpus.columns]
    if missing:
        raise ValueError(f"Phase 8 is missing columns: {missing}")

    corpus["record_id"] = corpus["record_id"].astype(str)
    dates = pd.to_datetime(
        corpus["created_utc"], errors="coerce", utc=True
    )
    corpus["audit_month"] = dates.dt.strftime("%Y-%m")

    feb_corpus = corpus[corpus["audit_month"] == "2016-02"].copy()

    print("Full Phase 8 records:", len(corpus))
    print("February 2016 records:", len(feb_corpus))
    print("\nFebruary corpus by community:")
    print(
        feb_corpus.groupby("community").size()
        .rename("phase8_records").to_string()
    )

    coverage_outputs = []
    prevalence_outputs = []
    monthly_topic_outputs = []

    for model_name, path in MODELS.items():
        print("\n" + "-" * 75)
        print("MODEL:", model_name)

        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")

        model = pd.read_csv(path, low_memory=False)

        if not {"record_id", "topic_label"}.issubset(model.columns):
            raise ValueError(
                f"Required columns missing in {path}: "
                f"{model.columns.tolist()}"
            )

        model["record_id"] = model["record_id"].astype(str)

        if model["record_id"].duplicated().any():
            raise ValueError(f"Duplicate record IDs in {model_name}.")

        unknown_ids = (
            set(model["record_id"]) - set(corpus["record_id"])
        )
        if unknown_ids:
            raise ValueError(
                f"{model_name} contains {len(unknown_ids)} unknown IDs."
            )

        feb_model = feb_corpus[
            ["record_id", "community", "audit_month"]
        ].merge(
            model[["record_id", "topic_label"]],
            on="record_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        feb_model["in_model_file"] = feb_model["_merge"].eq("both")
        feb_model["topic_label"] = feb_model["topic_label"].fillna(
            "OUTSIDE_MODEL_FILE"
        ).astype(str)

        upper = feb_model["topic_label"].str.strip().str.upper()
        feb_model["assigned"] = (
            feb_model["in_model_file"]
            & ~upper.eq("")
            & ~upper.isin(["NAN", "NONE", "MISSING_TOPIC"])
            & ~upper.str.contains(
                "UNASSIGNED|UNASSIGN|NO_TOPIC", regex=True, na=False
            )
        )

        feb_model["status"] = "ASSIGNED"
        feb_model.loc[
            feb_model["in_model_file"] & ~feb_model["assigned"],
            "status"
        ] = "UNASSIGNED_WITHIN_MODEL"
        feb_model.loc[
            ~feb_model["in_model_file"], "status"
        ] = "OUTSIDE_MODEL_FILE"

        # Coverage table: denominator is the complete February corpus
        # for each community.
        coverage = (
            feb_model.groupby("community")
            .agg(
                phase8_records=("record_id", "size"),
                records_in_model_file=("in_model_file", "sum"),
                assigned_records=("assigned", "sum"),
            )
            .reset_index()
        )

        coverage["unassigned_within_model_file"] = (
            coverage["records_in_model_file"]
            - coverage["assigned_records"]
        )
        coverage["outside_model_file"] = (
            coverage["phase8_records"]
            - coverage["records_in_model_file"]
        )
        coverage["model"] = model_name
        coverage["model_file_coverage_pct"] = (
            100 * coverage["records_in_model_file"]
            / coverage["phase8_records"]
        )
        coverage["assignment_rate_within_model_pct"] = (
            100 * coverage["assigned_records"]
            / coverage["records_in_model_file"].replace(0, pd.NA)
        )
        coverage["assigned_as_pct_of_phase8_pct"] = (
            100 * coverage["assigned_records"]
            / coverage["phase8_records"]
        )

        coverage_outputs.append(coverage)

        # Topic prevalence using assigned records as denominator.
        assigned = feb_model[feb_model["assigned"]].copy()

        prevalence = (
            assigned.groupby(["community", "topic_label"])
            .size()
            .rename("topic_records")
            .reset_index()
            .rename(columns={"topic_label": "topic"})
        )

        denominators = coverage[
            [
                "community", "phase8_records",
                "records_in_model_file", "assigned_records"
            ]
        ]

        prevalence = prevalence.merge(
            denominators, on="community", how="left"
        )
        prevalence["model"] = model_name

        prevalence["pct_among_assigned"] = (
            100 * prevalence["topic_records"]
            / prevalence["assigned_records"].replace(0, pd.NA)
        )
        prevalence["pct_of_model_file"] = (
            100 * prevalence["topic_records"]
            / prevalence["records_in_model_file"].replace(0, pd.NA)
        )
        prevalence["pct_of_phase8_community_corpus"] = (
            100 * prevalence["topic_records"]
            / prevalence["phase8_records"]
        )

        prevalence_outputs.append(prevalence)

        # Check monthly totals for the focal month.
        monthly_topics = (
            assigned.groupby(["audit_month", "topic_label"])
            .size()
            .rename("topic_records")
            .reset_index()
            .rename(columns={"topic_label": "topic"})
        )
        monthly_topics["model"] = model_name
        monthly_topic_outputs.append(monthly_topics)

        print("\nCommunity coverage:")
        print(coverage.to_string(index=False))

        print("\nTopic prevalence among assigned records:")
        print(
            prevalence[
                [
                    "community", "topic", "topic_records",
                    "assigned_records", "pct_among_assigned"
                ]
            ].to_string(index=False)
        )

    files = [
        (
            pd.concat(coverage_outputs, ignore_index=True),
            "phase10e_february_community_coverage.csv",
        ),
        (
            pd.concat(prevalence_outputs, ignore_index=True),
            "phase10e_february_community_topic_prevalence.csv",
        ),
        (
            pd.concat(monthly_topic_outputs, ignore_index=True),
            "phase10e_february_topic_totals_check.csv",
        ),
    ]

    for df, filename in files:
        target = OUT / filename
        df.to_csv(target, index=False)
        print("\nSaved:", target)
        print("Rows:", len(df))

    print("\n" + "=" * 75)
    print("PHASE 10E COMPLETED")
    print("Output directory:", OUT)
    print("Source files were not modified.")
    print("=" * 75)


if __name__ == "__main__":
    main()
