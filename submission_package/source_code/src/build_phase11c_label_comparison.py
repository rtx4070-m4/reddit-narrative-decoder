from pathlib import Path
import pandas as pd

source = Path(
    "results/tables/phase11a_consolidated_topic_dictionary/"
    "phase11a_consolidated_topic_dictionary.csv"
)
out_dir = Path("results/tables/phase11c_label_comparison")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source).fillna("")

pairs = [
    (
        "PAIR_1",
        ("NMF", "gaming", "Topic 4"),
        ("Semantic KMeans", "gaming", "Topic 2"),
        "Compare gaming character, gameplay, and narrative themes."
    ),
    (
        "PAIR_2",
        ("NMF", "gaming", "Topic 5"),
        ("NMF", "worldnews", "Topic 4"),
        "Check whether similar wording hides community-specific differences."
    ),
]

evidence_columns = [
    "working_topic_label",
    "label_status",
    "log_decision",
    "candidate_terms_from_sample",
    "representative_sample_texts",
    "supporting_evidence",
    "limitations_or_mixed_content",
    "evidence_suggestion_basis",
    "evidence_sampled_records",
    "evidence_nonempty_texts",
    "evidence_distinct_texts",
]

available = [c for c in evidence_columns if c in df.columns]
rows = []

for pair_id, key_a, key_b, question in pairs:
    for side, key in [("A", key_a), ("B", key_b)]:
        mask = (
            df["model"].astype(str).str.strip().eq(key[0])
            & df["community"].astype(str).str.strip().eq(key[1])
            & df["topic"].astype(str).str.strip().eq(key[2])
        )
        found = df.loc[mask]

        if len(found) != 1:
            raise ValueError(
                f"{pair_id} side {side}: expected one row for {key}, "
                f"found {len(found)}."
            )

        record = found.iloc[0]
        row = {
            "pair_id": pair_id,
            "side": side,
            "model": key[0],
            "community": key[1],
            "topic": key[2],
            "comparison_question": question,
        }
        for col in available:
            row[col] = record[col]
        rows.append(row)

review = pd.DataFrame(rows)
review["human_comparison_notes"] = ""
review["same_underlying_theme"] = ""
review["label_change_recommendation"] = ""

output = out_dir / "phase11c_label_evidence_comparison.csv"
review.to_csv(output, index=False)

print("=" * 72)
print("PHASE 11C — LABEL EVIDENCE COMPARISON")
print("=" * 72)
print(review[
    ["pair_id", "side", "model", "community", "topic",
     "working_topic_label", "log_decision"]
].to_string(index=False))

print("\nReview-sheet rows:", len(review))
print("Output:", output.resolve())
print("No source labels, assignments, or decisions were modified.")
print("Complete the human-review columns only after reading the evidence.")
