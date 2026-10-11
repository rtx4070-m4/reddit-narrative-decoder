from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "tables" / "phase8_cross_community"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "phase7b_reference": ROOT / "results" / "tables" / "nmf_document_topics_k8_v2.csv",
    "phase7e_topics": ROOT / "results" / "tables" / "phase7e_nmf" / "phase7e_document_topics_k8.csv",
    "phase7e_short": ROOT / "results" / "tables" / "phase7e_nmf" / "phase7e_excluded_short_documents.csv",
    "phase7e_unmodelled": ROOT / "results" / "tables" / "phase7e_nmf" / "phase7e_unmodelled_documents.csv",
    "phase8": OUT / "phase8_merged_observations.csv",
}


def read_csv(label, path):
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "record_id" not in df.columns:
        raise ValueError(
            f"{label} has no record_id column. Columns: {list(df.columns)}"
        )
    df["record_id"] = df["record_id"].astype("string").str.strip()
    return df


data = {name: read_csv(name, path) for name, path in FILES.items()}

sets = {
    name: set(df["record_id"].dropna().astype(str))
    for name, df in data.items()
}

ref_ids = sets["phase7b_reference"]
topic_ids = sets["phase7e_topics"]
short_ids = sets["phase7e_short"]
unmodelled_ids = sets["phase7e_unmodelled"]
phase8_ids = sets["phase8"]

missing_from_topics = ref_ids - topic_ids
short_union_unmodelled = short_ids | unmodelled_ids
overlap_short_unmodelled = short_ids & unmodelled_ids

print("=== Phase 7E exclusion reconciliation ===")
for name, df in data.items():
    print(f"{name}: rows={len(df):,}; unique IDs={len(sets[name]):,}")

print(f"\nPhase 7B reference IDs absent from Phase 7E topic output: {len(missing_from_topics):,}")
print(f"Short-exclusion IDs: {len(short_ids):,}")
print(f"Unmodelled IDs: {len(unmodelled_ids):,}")
print(f"IDs appearing in both exclusion files: {len(overlap_short_unmodelled):,}")
print(f"Unique IDs across both exclusion files: {len(short_union_unmodelled):,}")

print("\nSet checks:")
print(f"Missing reference IDs covered by exclusion union: {len(missing_from_topics & short_union_unmodelled):,}")
print(f"Missing reference IDs NOT in either exclusion file: {len(missing_from_topics - short_union_unmodelled):,}")
print(f"Exclusion IDs that were not in missing reference IDs: {len(short_union_unmodelled - missing_from_topics):,}")
print(f"Phase 8 IDs absent from Phase 7B reference: {len(phase8_ids - ref_ids):,}")
print(f"Phase 7B reference IDs absent from Phase 8: {len(ref_ids - phase8_ids):,}")

# Build an ID-level report for all Phase 7B IDs, indicating the outcome.
ref = data["phase7b_reference"].copy()
topic_id_set = sets["phase7e_topics"]
short_id_set = sets["phase7e_short"]
unmodelled_id_set = sets["phase7e_unmodelled"]

def classify(record_id):
    rid = str(record_id)
    if rid in topic_id_set:
        return "present_in_phase7e_topic_output"
    if rid in short_id_set and rid in unmodelled_id_set:
        return "listed_in_both_exclusion_files"
    if rid in short_id_set:
        return "excluded_short_text"
    if rid in unmodelled_id_set:
        return "eligible_but_no_retained_tfidf_features"
    return "missing_reason_not_found_in_exclusion_files"

ref["phase7e_reconciliation_status"] = ref["record_id"].map(classify)
ref.to_csv(OUT / "phase7e_id_exclusion_reconciliation.csv", index=False)

# Save all Phase 8 records that never entered the Phase 7B reference.
phase8 = data["phase8"]
phase8_missing_ref = phase8[
    phase8["record_id"].astype(str).isin(phase8_ids - ref_ids)
].copy()
phase8_missing_ref.to_csv(
    OUT / "phase8_ids_absent_from_phase7b_reference.csv", index=False
)

print("\nReconciliation status counts:")
print(ref["phase7e_reconciliation_status"].value_counts(dropna=False).to_string())
print(f"\nSaved detailed reports in: {OUT}")
