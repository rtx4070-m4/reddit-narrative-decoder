from pathlib import Path
import pandas as pd

root = Path.cwd()
base = root / "results/tables"

revision_path = base / "phase10s_label_revision/phase10s_group_revision_sheet.csv"
log_path = base / "phase10o_human_validation/phase10_review_decision_log.csv"

keys = ["model", "community", "topic"]
revisions = pd.read_csv(revision_path, dtype=str, keep_default_na=False)
log = pd.read_csv(log_path, dtype=str, keep_default_na=False)

print("=" * 65)
print("PHASE 10 FINAL INTEGRITY AUDIT")
print("=" * 65)

required = keys + ["decision", "reviewer_notes"]
missing_columns = [c for c in required if c not in log.columns]
if missing_columns:
    print("FAIL: Missing decision-log columns:", missing_columns)
    raise SystemExit(1)

expected = set(
    tuple(row[k] for k in keys)
    for _, row in revisions[keys].drop_duplicates().iterrows()
)
actual = set(
    tuple(row[k] for k in keys)
    for _, row in log.iterrows()
)

duplicates = log.duplicated(keys, keep=False)
invalid_decisions = ~log["decision"].isin(["accept", "revise", "do_not_label"])
missing_groups = expected - actual
unexpected_groups = actual - expected
batch_approved = log["reviewer_notes"].str.contains(
    "BATCH APPROVED", case=False, na=False
)

print("Expected unique topic groups:", len(expected))
print("Decision-log records:", len(log))
print("Unique groups recorded:", len(actual))
print("Duplicate group rows:", int(duplicates.sum()))
print("Missing expected groups:", len(missing_groups))
print("Unexpected groups:", len(unexpected_groups))
print("Invalid decision values:", int(invalid_decisions.sum()))
print("Records marked as batch-approved:", int(batch_approved.sum()))

print("\nDecision counts:")
print(log["decision"].value_counts(dropna=False).to_string())

if missing_groups:
    print("\nMissing groups:")
    for item in sorted(missing_groups):
        print(item)

if unexpected_groups:
    print("\nUnexpected groups:")
    for item in sorted(unexpected_groups):
        print(item)

passed = (
    len(expected) == 40
    and len(log) == 40
    and len(actual) == 40
    and not duplicates.any()
    and not invalid_decisions.any()
    and not missing_groups
    and not unexpected_groups
)

print("\n" + ("AUDIT PASSED" if passed else "AUDIT NEEDS ATTENTION"))
print("Original review files were not rewritten by this audit.")
print("This audit checks the current records; it cannot independently prove historical file immutability.")
