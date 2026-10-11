from pathlib import Path
import pandas as pd

BASE = Path("results/tables")

FILES = {
    "dictionary": BASE / "phase10i_topic_dictionary" / "phase10i_provisional_topic_dictionary.csv",
    "revision": BASE / "phase10s_label_revision" / "phase10s_group_revision_sheet.csv",
    "evidence": BASE / "phase10p_review_pack" / "phase10p_group_evidence_review.csv",
    "decision_log": BASE / "phase10o_human_validation" / "phase10_review_decision_log.csv",
}

OUT = BASE / "phase11a_consolidated_topic_dictionary"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["model", "community", "topic"]

def load_csv(name, path):
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")

    for col in KEYS:
        df[col] = df[col].astype("string").str.strip()

    # Normalize only the known model-name variation.
    df["model"] = df["model"].replace({
        "Semantic_KMeans": "Semantic KMeans"
    })

    if df[KEYS].isna().any().any() or df[KEYS].eq("").any().any():
        raise ValueError(f"{name} contains blank key values.")

    duplicates = df.duplicated(KEYS, keep=False)
    if duplicates.any():
        print(f"\nDuplicate keys in {name}:")
        print(df.loc[duplicates, KEYS].to_string(index=False))
        raise ValueError(f"{name} contains duplicate group keys.")

    return df

dictionary = load_csv("dictionary", FILES["dictionary"])
revision = load_csv("revision", FILES["revision"])
evidence = load_csv("evidence", FILES["evidence"])
decision_log = load_csv("decision_log", FILES["decision_log"])

dict_keys = set(map(tuple, dictionary[KEYS].to_numpy()))
for name, frame in [
    ("revision", revision),
    ("evidence", evidence),
    ("decision_log", decision_log),
]:
    other_keys = set(map(tuple, frame[KEYS].to_numpy()))
    missing = sorted(dict_keys - other_keys)
    extra = sorted(other_keys - dict_keys)

    print(f"{name}: missing={len(missing)}, extra={len(extra)}")
    if missing or extra:
        raise ValueError(
            f"Key mismatch remains for {name}; integration stopped safely."
        )

def select_with_prefix(frame, prefix, exclude_keys=True):
    cols = [c for c in frame.columns if not (exclude_keys and c in KEYS)]
    selected = frame[KEYS + cols].copy()
    return selected.rename(columns={
        c: f"{prefix}_{c}" for c in cols
    })

result = dictionary.copy()
result = result.merge(
    select_with_prefix(revision, "revision"),
    on=KEYS, how="left", validate="one_to_one", indicator="_revision_merge"
)
result = result.merge(
    select_with_prefix(evidence, "evidence"),
    on=KEYS, how="left", validate="one_to_one", indicator="_evidence_merge"
)
result = result.merge(
    select_with_prefix(decision_log, "log"),
    on=KEYS, how="left", validate="one_to_one", indicator="_log_merge"
)

result["working_topic_label"] = (
    result.get("revision_suggested_label", pd.Series("", index=result.index))
    .fillna("").astype(str).str.strip()
)
if "proposed_topic_label" in result.columns:
    blank = result["working_topic_label"].eq("")
    result.loc[blank, "working_topic_label"] = (
        result.loc[blank, "proposed_topic_label"].fillna("").astype(str).str.strip()
    )

result["decision_record_status"] = result["_log_merge"].map({
    "both": "matched_decision_log",
    "left_only": "no_matching_decision_log",
})

decision = result.get("log_decision", pd.Series("", index=result.index))
decision = decision.fillna("").astype(str).str.strip().str.lower()

notes = result.get("log_reviewer_notes", pd.Series("", index=result.index))
notes = notes.fillna("").astype(str).str.lower()
batch = notes.str.contains("batch approved", regex=False)

result["label_status"] = "recorded_separate_decision"
result.loc[result["_log_merge"].ne("both"), "label_status"] = "no_matching_decision_log"
result.loc[batch & decision.eq("accept"), "label_status"] = "batch_approved_provisional"
result.loc[batch & decision.eq("do_not_label"), "label_status"] = "batch_approved_recommendation"
result.loc[~batch & decision.eq("do_not_label"), "label_status"] = "recorded_do_not_label"
result.loc[~batch & decision.eq("accept"), "label_status"] = "recorded_accept"

warnings = []
for _, row in result.iterrows():
    row_warnings = []
    if not str(row["working_topic_label"]).strip():
        row_warnings.append("missing_working_label")
    if row["label_status"] in {
        "recorded_do_not_label",
        "batch_approved_recommendation",
    }:
        row_warnings.append("Do not interpret this topic as having an accepted label")
    warnings.append("; ".join(row_warnings))

result["integration_warning"] = warnings

result = result.drop(columns=[
    c for c in ["_revision_merge", "_evidence_merge", "_log_merge"]
    if c in result.columns
])

out_file = OUT / "phase11a_consolidated_topic_dictionary.csv"
audit_file = OUT / "phase11a_integration_audit.csv"

result.to_csv(out_file, index=False)

audit = pd.DataFrame([
    {"check": "dictionary_rows", "value": len(dictionary)},
    {"check": "revision_groups", "value": len(revision)},
    {"check": "evidence_groups", "value": len(evidence)},
    {"check": "decision_log_groups", "value": len(decision_log)},
    {"check": "output_rows", "value": len(result)},
    {"check": "unique_output_groups", "value": result[KEYS].drop_duplicates().shape[0]},
    {"check": "dictionary_groups_missing_revision",
     "value": len(dict_keys - set(map(tuple, revision[KEYS].to_numpy())))},
    {"check": "dictionary_groups_missing_evidence",
     "value": len(dict_keys - set(map(tuple, evidence[KEYS].to_numpy())))},
    {"check": "dictionary_groups_missing_decision",
     "value": len(dict_keys - set(map(tuple, decision_log[KEYS].to_numpy())))},
    {"check": "revision_groups_not_in_dictionary",
     "value": len(set(map(tuple, revision[KEYS].to_numpy())) - dict_keys)},
    {"check": "decision_groups_not_in_dictionary",
     "value": len(set(map(tuple, decision_log[KEYS].to_numpy())) - dict_keys)},
    {"check": "rows_with_missing_working_label",
     "value": int(result["working_topic_label"].fillna("").str.strip().eq("").sum())},
    {"check": "rows_with_integration_warning",
     "value": int(result["integration_warning"].fillna("").str.strip().ne("").sum())},
    {"check": "batch_approved_provisional_rows",
     "value": int(result["label_status"].eq("batch_approved_provisional").sum())},
    {"check": "batch_approved_recommendation_rows",
     "value": int(result["label_status"].eq("batch_approved_recommendation").sum())},
    {"check": "recorded_do_not_label_rows",
     "value": int(result["label_status"].eq("recorded_do_not_label").sum())},
])

audit.to_csv(audit_file, index=False)

print("=" * 72)
print("PHASE 11A — CONSOLIDATED TOPIC DICTIONARY")
print("=" * 72)
print(f"Dictionary rows: {len(dictionary)}")
print(f"Integrated rows: {len(result)}")
print(f"Unique output groups: {result[KEYS].drop_duplicates().shape[0]}")
print(f"Rows with missing working labels: {audit.loc[audit['check'].eq('rows_with_missing_working_label'), 'value'].iloc[0]}")
print(f"Rows with integration warnings: {audit.loc[audit['check'].eq('rows_with_integration_warning'), 'value'].iloc[0]}")

print("\nLabel-status counts:")
print(result["label_status"].value_counts(dropna=False).to_string())

print("\nDecision counts:")
print(result.get("log_decision", pd.Series(dtype="object")).value_counts(dropna=False).to_string())

print("\nAudit checks:")
print(audit.to_string(index=False))

print("\nOutput files:")
print(out_file.resolve())
print(audit_file.resolve())
print("\nOriginal Phase 7 and Phase 10 source files were not modified.")
print("Integration does not independently validate topic meaning.")
