from pathlib import Path
import pandas as pd

root = Path.cwd()
base = root / "results/tables"

revision_path = base / "phase10s_label_revision/phase10s_group_revision_sheet.csv"
evidence_path = base / "phase10p_review_pack/phase10p_group_evidence_review.csv"
examples_path = base / "phase10o_human_validation/phase10o_example_rating_sheet.csv"
log_path = base / "phase10o_human_validation/phase10_review_decision_log.csv"
out_path = base / "phase10o_human_validation/phase10_batch_review_pending.csv"

revisions = pd.read_csv(revision_path)
evidence = pd.read_csv(evidence_path)
examples = pd.read_csv(examples_path)

keys = ["model", "community", "topic"]

if log_path.exists():
    log = pd.read_csv(log_path, dtype=str, keep_default_na=False)
else:
    log = pd.DataFrame(columns=keys + ["decision"])

completed = set()
if not log.empty:
    for _, row in log.iterrows():
        if row.get("decision", "") in {"accept", "revise", "do_not_label"}:
            completed.add(tuple(str(row[k]) for k in keys))

for name, frame in [
    ("revision", revisions),
    ("evidence", evidence),
    ("examples", examples),
]:
    missing = [k for k in keys if k not in frame.columns]
    if missing:
        raise ValueError(f"{name} file missing group keys: {missing}")

# Build one row per group from the existing review evidence.
batch = revisions.drop_duplicates(keys).copy()

evidence_columns = [
    c for c in [
        "proposed_label", "overall_coherence", "label_confidence",
        "suggested_decision", "justification", "suggestion_basis",
        "review_priority", "sampled_records", "nonempty_texts",
        "distinct_texts"
    ]
    if c in evidence.columns and c not in keys
]

batch = batch.merge(
    evidence[keys + evidence_columns].drop_duplicates(keys),
    on=keys,
    how="left",
    suffixes=("", "_evidence")
)

batch["existing_decision_status"] = batch.apply(
    lambda r: (
        "already_confirmed"
        if tuple(str(r[k]) for k in keys) in completed
        else "pending_batch_approval"
    ),
    axis=1
)

# Include the actual sample texts for one-pass inspection.
def collect_examples(group):
    subset = examples
    for key in keys:
        subset = subset[subset[key].astype(str) == str(group[key])]
    texts = subset["review_text"].fillna("").astype(str).tolist()
    return "\n---\n".join(
        f"{i}. {text[:600]}" for i, text in enumerate(texts, 1)
    )

batch["sampled_examples_text"] = batch.apply(collect_examples, axis=1)

# No decisions are made or written to the human decision log here.
batch["batch_human_decision"] = ""
batch["batch_revised_label"] = ""
batch["batch_reviewer_notes"] = ""

out_path.parent.mkdir(parents=True, exist_ok=True)
batch.to_csv(out_path, index=False, encoding="utf-8")

print("=" * 72)
print("BATCH REVIEW PACKAGE CREATED")
print("=" * 72)
print("Groups in package:", len(batch))
print("Already confirmed:", int((batch["existing_decision_status"] == "already_confirmed").sum()))
print("Pending approval:", int((batch["existing_decision_status"] == "pending_batch_approval").sum()))
print("Original review spreadsheets modified: NO")
print("Human decision log modified: NO")
print("AI API called: NO")
print("\nOutput:")
print(out_path)

print("\nEXISTING SUGGESTED DECISIONS")
if "suggested_decision" in batch.columns:
    print(batch["suggested_decision"].fillna("(missing)").value_counts().to_string())
else:
    print("No suggested_decision column available.")

print("\nPRIORITY GROUPS")
cols = [c for c in [
    "review_rank", "model", "community", "topic",
    "suggested_label", "proposed_label",
    "suggested_decision", "label_confidence",
    "existing_decision_status"
] if c in batch.columns]
print(batch[cols].sort_values(
    "review_rank" if "review_rank" in cols else cols[0],
    kind="stable"
).head(12).to_string(index=False))

print("\nThis is a review package, not a set of confirmed human decisions.")
