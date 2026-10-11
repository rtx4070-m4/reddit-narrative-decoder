from pathlib import Path
import pandas as pd

base = Path("results/tables/phase7o_temporal_narratives_v17")

models = {
    "NMF": (
        base / "nmf_document_topic_date_join.csv",
        base / "nmf_community_month_topic_prevalence.csv",
    ),
    "Semantic_KMeans": (
        base / "semantic_kmeans_document_topic_date_join.csv",
        base / "semantic_kmeans_community_month_topic_prevalence.csv",
    ),
}

all_passed = True

for model, (doc_path, table_path) in models.items():
    print("\n" + "=" * 70)
    print(model, "FULL COMMUNITY-MONTH VALIDATION")
    print("=" * 70)

    docs = pd.read_csv(doc_path)
    table = pd.read_csv(table_path)

    docs["month"] = docs["month"].astype(str)
    table["month"] = table["month"].astype(str)

    # Include UNASSIGNED records in the expected counts.
    expected_counts = (
        docs.groupby(["community", "month", "topic_label"], dropna=False)
        .size()
        .rename("expected_count")
        .reset_index()
    )

    actual_counts = table[
        ["community", "month", "topic_label", "document_count"]
    ]

    counts = expected_counts.merge(
        actual_counts,
        on=["community", "month", "topic_label"],
        how="outer",
        indicator=True,
    )

    counts["expected_count"] = counts["expected_count"].fillna(0)
    counts["document_count"] = counts["document_count"].fillna(0)

    count_mismatch = (
        counts["expected_count"].astype(int)
        != counts["document_count"].astype(int)
    )
    count_keys_missing = counts["_merge"].ne("both")

    expected_den = (
        docs.assign(
            assigned=docs["topic_label"].ne("UNASSIGNED")
        )
        .groupby(["community", "month"])
        .agg(
            expected_all=("record_id", "size"),
            expected_assigned=("assigned", "sum"),
        )
        .reset_index()
    )

    actual_den = table[
        [
            "community", "month",
            "community_month_total_all_rows",
            "community_month_total_assigned",
        ]
    ].drop_duplicates()

    den = expected_den.merge(
        actual_den,
        on=["community", "month"],
        how="outer",
        indicator=True,
    )

    denominator_mismatch = (
        den["_merge"].ne("both")
        | den["expected_all"].fillna(-1).ne(
            den["community_month_total_all_rows"].fillna(-2)
        )
        | den["expected_assigned"].fillna(-1).ne(
            den["community_month_total_assigned"].fillna(-2)
        )
    )

    all_share_expected = (
        table["document_count"]
        / table["community_month_total_all_rows"]
    )
    assigned_share_expected = (
        table["document_count"]
        / table["community_month_total_assigned"]
    )

    all_share_mismatch = (
        table["share_of_all_community_month_rows"] - all_share_expected
    ).abs() > 1e-8
    assigned_share_mismatch = (
        table["share_among_assigned_rows"] - assigned_share_expected
    ).abs() > 1e-8

    print("Source rows:", len(docs))
    print("Prevalence rows:", len(table))
    print("Topic-count mismatches:", int(count_mismatch.sum()))
    print("Missing/extra topic keys:", int(count_keys_missing.sum()))
    print("Denominator group mismatches:", int(denominator_mismatch.sum()))
    print("All-record proportion mismatches:", int(all_share_mismatch.sum()))
    print("Assigned-record proportion mismatches:", int(assigned_share_mismatch.sum()))

    passed = not (
        count_mismatch.any()
        or count_keys_missing.any()
        or denominator_mismatch.any()
        or all_share_mismatch.any()
        or assigned_share_mismatch.any()
    )

    print("RESULT:", "PASS" if passed else "FAIL")
    all_passed = all_passed and passed

    if not passed:
        if count_mismatch.any() or count_keys_missing.any():
            print("\nTopic-count discrepancies:")
            print(counts.loc[count_mismatch | count_keys_missing].to_string(index=False))
        if denominator_mismatch.any():
            print("\nDenominator discrepancies:")
            print(den.loc[denominator_mismatch].to_string(index=False))

print("\nOVERALL RESULT:", "PASS" if all_passed else "FAIL")
print("Existing analysis result files were not modified.")
