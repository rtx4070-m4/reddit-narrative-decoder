from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

ROOT = Path.cwd()
EXAMPLES_FILE = ROOT / "results/tables/phase10o_human_validation/phase10o_example_rating_sheet.csv"
REVISION_FILE = ROOT / "results/tables/phase10s_label_revision/phase10s_group_revision_sheet.csv"
LOG_FILE = ROOT / "results/tables/phase10o_human_validation/phase10_review_decision_log.csv"

KEYS = ["model", "community", "topic"]
LOG_COLUMNS = [
    "model", "community", "topic", "decision", "final_label",
    "human_confidence", "evidence_for", "evidence_against",
    "reviewer_notes", "reviewed_at_utc"
]


def load_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def get_priority(group):
    rank = pd.to_numeric(
        pd.Series([group.get("review_rank")]), errors="coerce"
    ).iloc[0]
    return rank if pd.notna(rank) else float("inf")


def main():
    examples = load_csv(EXAMPLES_FILE)
    revisions = load_csv(REVISION_FILE)

    for name, frame in [("examples", examples), ("revisions", revisions)]:
        missing = [c for c in KEYS if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    if "review_text" not in examples.columns:
        raise ValueError("Example file does not contain review_text.")
    if "suggested_label" not in revisions.columns:
        raise ValueError("Revision file does not contain suggested_label.")

    groups = revisions.drop_duplicates(KEYS).copy()
    groups["_rank"] = groups.apply(get_priority, axis=1)
    groups = groups.sort_values("_rank", kind="stable")

    # The separate log is the sole source for determining completed reviews.
    if LOG_FILE.exists():
        log = pd.read_csv(LOG_FILE, dtype=str, keep_default_na=False)
        if list(log.columns) != LOG_COLUMNS:
            raise ValueError(
                "Decision-log schema differs from the expected schema. "
                "No changes were made."
            )
    else:
        log = pd.DataFrame(columns=LOG_COLUMNS)

    completed = set()
    for _, row in log.iterrows():
        if row["decision"] in {"accept", "revise", "do_not_label"}:
            completed.add(tuple(row[k] for k in KEYS))

    remaining = groups[
        ~groups.apply(
            lambda r: tuple(str(r[k]) for k in KEYS) in completed,
            axis=1
        )
    ]

    print("\n" + "=" * 72)
    print("REDDIT NARRATIVE DECODER — LOCAL HUMAN REVIEW")
    print("=" * 72)
    print("Total topic groups:", len(groups))
    print("Groups with confirmed decisions:", len(completed))
    print("Groups remaining:", len(remaining))
    print("AI API: disabled")
    print("Original spreadsheets: read-only")

    if remaining.empty:
        print("\nAll topic groups have confirmed decisions in the separate log.")
        print("Run the audit step to validate the saved decisions.")
        return

    group = remaining.iloc[0]
    key = tuple(str(group[k]) for k in KEYS)

    print("\nNEXT GROUP")
    for k in KEYS:
        print(f"{k}: {group[k]}")
    print("Suggested label:", group["suggested_label"])
    for col in ["review_rank", "revision_priority", "label_confidence"]:
        if col in group.index and pd.notna(group[col]):
            print(f"{col}: {group[col]}")

    mask = pd.Series(True, index=examples.index)
    for k in KEYS:
        mask &= examples[k].astype(str).eq(key[KEYS.index(k)])
    sample = examples.loc[mask]

    print(f"\nSAMPLED EXAMPLES: {len(sample)}")
    print("-" * 72)
    for i, text in enumerate(sample["review_text"].fillna(""), 1):
        print(f"{i}. {str(text)[:700]}")

    print("\nDECISION OPTIONS")
    print("  1 = accept suggested label")
    print("  2 = revise label")
    print("  3 = do_not_label")
    print("  4 = skip for now (not recorded as a decision)")
    choice = input("Your choice [1-4]: ").strip()

    if choice == "4":
        print("Skipped. No decision was saved.")
        return
    if choice not in {"1", "2", "3"}:
        print("Invalid choice. No decision was saved.")
        return

    proposed = str(group["suggested_label"]).strip()
    decision = {"1": "accept", "2": "revise", "3": "do_not_label"}[choice]
    final_label = proposed if decision == "accept" else ""

    if decision == "revise":
        final_label = input("Enter your revised topic label: ").strip()
        if not final_label:
            print("A revised label is required. No decision was saved.")
            return
    elif decision == "do_not_label":
        print("The final label will be blank for this decision.")

    confidence = input("Your confidence [high/moderate/low] (optional): ").strip().lower()
    if confidence and confidence not in {"high", "moderate", "low"}:
        print("Invalid confidence. No decision was saved.")
        return

    evidence_for = input("Evidence supporting your decision (optional): ").strip()
    evidence_against = input("Contradictory evidence (optional): ").strip()
    notes = input("Reviewer notes (optional): ").strip()

    print("\nCONFIRM DECISION")
    print("Group:", " | ".join(key))
    print("Decision:", decision)
    print("Final label:", final_label or "(no label)")
    confirm = input("Save this human decision? Type YES to confirm: ").strip()

    if confirm != "YES":
        print("Not confirmed. Nothing was saved.")
        return

    # Re-read the log before appending to reduce accidental duplicate decisions.
    if LOG_FILE.exists():
        current = pd.read_csv(LOG_FILE, dtype=str, keep_default_na=False)
    else:
        current = pd.DataFrame(columns=LOG_COLUMNS)

    if not current.empty:
        already = current.apply(
            lambda r: all(str(r[k]) == key[i] for i, k in enumerate(KEYS))
            and r["decision"] in {"accept", "revise", "do_not_label"},
            axis=1
        ).any()
        if already:
            print("This group already has a confirmed decision. Nothing appended.")
            return

    record = {
        "model": key[0],
        "community": key[1],
        "topic": key[2],
        "decision": decision,
        "final_label": final_label,
        "human_confidence": confidence,
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
        "reviewer_notes": notes,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat()
    }

    updated = pd.concat([current, pd.DataFrame([record])], ignore_index=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(LOG_FILE, index=False, encoding="utf-8")

    print("\nHuman decision saved to:")
    print(LOG_FILE)
    print("Original review spreadsheets were not modified.")


if __name__ == "__main__":
    main()
