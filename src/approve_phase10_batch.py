from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

root = Path.cwd()
base = root / "results/tables"

revision_path = base / "phase10s_label_revision/phase10s_group_revision_sheet.csv"
evidence_path = base / "phase10p_review_pack/phase10p_group_evidence_review.csv"
log_path = base / "phase10o_human_validation/phase10_review_decision_log.csv"

keys = ["model", "community", "topic"]
columns = [
    "model", "community", "topic", "decision", "final_label",
    "human_confidence", "evidence_for", "evidence_against",
    "reviewer_notes", "reviewed_at_utc"
]

revisions = pd.read_csv(revision_path, dtype=str, keep_default_na=False)
evidence = pd.read_csv(evidence_path, dtype=str, keep_default_na=False)

if log_path.exists():
    log = pd.read_csv(log_path, dtype=str, keep_default_na=False)
    if list(log.columns) != columns:
        raise ValueError("Decision-log columns differ from expected schema. Stopped.")
else:
    log = pd.DataFrame(columns=columns)

for name, frame in [("revision", revisions), ("evidence", evidence)]:
    missing = [k for k in keys if k not in frame.columns]
    if missing:
        raise ValueError(f"{name} file is missing keys: {missing}")

if "suggested_label" not in revisions.columns:
    raise ValueError("No suggested_label column found. Stopped.")

if "suggested_decision" not in evidence.columns:
    raise ValueError("No suggested_decision column found. Stopped.")

recommendations = revisions.drop_duplicates(keys).merge(
    evidence[keys + [
        "suggested_decision", "suggested_confidence", "suggestion_basis"
    ]].drop_duplicates(keys),
    on=keys,
    how="left",
    validate="one_to_one"
)

completed = set()
for _, row in log.iterrows():
    if row.get("decision", "") in {"accept", "revise", "do_not_label"}:
        completed.add(tuple(row[k] for k in keys))

pending = recommendations[
    ~recommendations.apply(
        lambda r: tuple(str(r[k]) for k in keys) in completed,
        axis=1
    )
].copy()

print("=" * 72)
print("BATCH APPROVAL — PREVIEW")
print("=" * 72)
print("Already recorded:", len(completed))
print("Pending recommendations:", len(pending))
print("Original review spreadsheets will remain untouched.")

if pending.empty:
    print("No pending recommendations.")
    raise SystemExit(0)

print("\nRecommendations to be recorded:")
for _, row in pending.iterrows():
    print(
        f"- {row['model']} | {row['community']} | {row['topic']} "
        f"| label={row['suggested_label']} "
        f"| recommendation={row['suggested_decision']}"
    )

print("\nThis is a batch approval, NOT an individual review of every example.")
print("Provisional recommendations will remain marked provisional in reviewer notes.")
print("Only approve if you accept this limitation.")

expected = f"APPROVE ALL {len(pending)} RECOMMENDATIONS"
answer = input(f"\nType exactly '{expected}' to proceed: ").strip()

if answer != expected:
    print("Confirmation did not match. Nothing was saved.")
    raise SystemExit(0)

now = datetime.now(timezone.utc).isoformat()
new_rows = []

for _, row in pending.iterrows():
    recommendation = str(row["suggested_decision"]).strip().lower()
    label = str(row["suggested_label"]).strip()

    if recommendation == "do_not_label":
        decision = "do_not_label"
        final_label = ""
    elif recommendation == "provisional" and label:
        decision = "accept"
        final_label = label
    else:
        print(
            "STOP: Unsupported recommendation or missing label for",
            tuple(row[k] for k in keys)
        )
        print("No rows were saved. Resolve this group separately.")
        raise SystemExit(1)

    confidence = str(row.get("suggested_confidence", "")).strip().lower()
    if confidence not in {"high", "moderate", "low"}:
        confidence = ""

    basis = str(row.get("suggestion_basis", "")).strip()
    notes = (
        f"BATCH APPROVED by explicit user confirmation. "
        f"Existing recommendation: {recommendation}. "
        f"This was not an individual example-by-example human review. "
        f"Original recommendation retained as provisional where applicable. "
        f"Basis: {basis}"
    )

    new_rows.append({
        "model": str(row["model"]),
        "community": str(row["community"]),
        "topic": str(row["topic"]),
        "decision": decision,
        "final_label": final_label,
        "human_confidence": confidence,
        "evidence_for": "",
        "evidence_against": "",
        "reviewer_notes": notes,
        "reviewed_at_utc": now
    })

updated = pd.concat(
    [log, pd.DataFrame(new_rows, columns=columns)],
    ignore_index=True
)

if updated.duplicated(keys).any():
    raise ValueError("Duplicate group decisions detected. Nothing saved.")

log_path.parent.mkdir(parents=True, exist_ok=True)
updated.to_csv(log_path, index=False, encoding="utf-8")

print("\nBatch approval saved.")
print("New records:", len(new_rows))
print("Total recorded groups:", len(updated))
print("Decision log:", log_path)
print("Original review spreadsheets modified: NO")
print("Individual example-by-example review falsely claimed: NO")
