from pathlib import Path
import pandas as pd

base = Path("results/tables/phase7o_temporal_narratives_v17")

files = {
    "NMF": {
        "documents": base / "nmf_document_topic_date_join.csv",
        "monthly": base / "nmf_monthly_topic_prevalence.csv",
        "community_monthly": base / "nmf_community_month_topic_prevalence.csv",
    },
    "Semantic_KMeans": {
        "documents": base / "semantic_kmeans_document_topic_date_join.csv",
        "monthly": base / "semantic_kmeans_monthly_topic_prevalence.csv",
        "community_monthly": base / "semantic_kmeans_community_month_topic_prevalence.csv",
    },
}

for model, paths in files.items():
    print("\n" + "=" * 75)
    print(f"{model}: MONTHLY PREVALENCE RECONCILIATION")
    print("=" * 75)

    docs_path = paths["documents"]
    if not docs_path.exists():
        print("Missing document file:", docs_path)
        continue

    docs = pd.read_csv(docs_path)
    docs["month"] = docs["month"].astype(str)
    docs["is_unassigned"] = docs["topic_label"].eq("UNASSIGNED")

    print("\nDocument-level checks")
    print("Total rows:", len(docs))
    print("Assigned by topic label:", int((~docs["is_unassigned"]).sum()))
    print("Unassigned by topic label:", int(docs["is_unassigned"].sum()))

    if "has_assignment" in docs.columns:
        flags = docs["has_assignment"].astype(str).str.strip().str.lower()
        flag_assigned = flags.isin(["true", "1", "yes"])
        print("Assignment flag says assigned:", int(flag_assigned.sum()))
        print(
            "Flag/topic-label disagreements:",
            int((flag_assigned != ~docs["is_unassigned"]).sum())
        )

    expected = (
        docs.groupby(["month", "topic_label"], dropna=False)
        .size()
        .rename("document_count")
        .reset_index()
    )
    expected_months = (
        docs.groupby("month")
        .agg(
            source_rows=("record_id", "size"),
            assigned_rows=("is_unassigned", lambda x: int((~x).sum())),
            unassigned_rows=("is_unassigned", "sum"),
        )
        .reset_index()
    )

    for kind in ["monthly", "community_monthly"]:
        path = paths[kind]
        print(f"\n--- {kind} file ---")

        if not path.exists():
            print("MISSING:", path)
            continue

        table = pd.read_csv(path)
        print("Path:", path)
        print("Rows:", len(table))
        print("Columns:", list(table.columns))
        print("First 5 rows:")
        print(table.head(5).to_string(index=False))

        if kind == "monthly":
            required = {"month", "topic_label", "document_count"}
            if required.issubset(table.columns):
                actual = table[["month", "topic_label", "document_count"]].copy()
                actual["month"] = actual["month"].astype(str)

                merged = expected.merge(
                    actual,
                    on=["month", "topic_label"],
                    how="outer",
                    suffixes=("_expected", "_actual"),
                ).fillna(0)

                mismatch = (
                    merged["document_count_expected"].astype(int)
                    != merged["document_count_actual"].astype(int)
                )
                print("Monthly topic-count mismatches:", int(mismatch.sum()))
                if mismatch.any():
                    print(merged.loc[mismatch].to_string(index=False))
            else:
                print("Could not compare counts: required columns are absent.")

            denominator_cols = [
                c for c in table.columns
                if any(s in c.lower() for s in
                       ["month_total", "share_", "denominator", "assigned_rows"])
            ]
            if denominator_cols:
                print("Denominator/proportion columns:", denominator_cols)

        else:
            print(
                "Community-month table requires a separate grouped comparison "
                "if its schema differs from the monthly table."
            )

    print("\nExpected monthly source totals:")
    print(expected_months.to_string(index=False))

