from pathlib import Path
from difflib import SequenceMatcher
import re
import pandas as pd

SOURCE = Path(
    "results/tables/phase11a_consolidated_topic_dictionary/"
    "phase11a_consolidated_topic_dictionary.csv"
)

OUT = Path("results/tables/phase11b_topic_quality_audit")
OUT.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(f"Phase 11A dictionary not found: {SOURCE}")

df = pd.read_csv(SOURCE).fillna("")

required = [
    "model", "community", "topic", "working_topic_label",
    "label_status", "log_decision"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Required columns missing: {missing}")

def clean(value):
    return re.sub(r"\s+", " ", str(value).strip())

def normalize_label(value):
    value = clean(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()

def nonempty(value):
    return bool(clean(value))

# One row per topic group, retaining the source label and decision.
audit = df.copy()

audit["label_normalized"] = audit["working_topic_label"].map(normalize_label)
audit["label_missing"] = audit["label_normalized"].eq("")
audit["label_short_or_generic"] = audit["label_normalized"].map(
    lambda x: len(x.split()) < 3 if x else True
)

audit["is_do_not_label"] = (
    audit["log_decision"].astype(str).str.strip().str.lower()
    .eq("do_not_label")
)

audit["is_batch_approved"] = (
    audit["label_status"].astype(str).str.contains(
        "batch_approved", case=False, regex=False
    )
)

# Detect duplicate normalized labels across groups.
label_counts = audit.loc[
    audit["label_normalized"].ne(""), "label_normalized"
].value_counts()

audit["exact_duplicate_label"] = audit["label_normalized"].map(
    lambda x: label_counts.get(x, 0) > 1 if x else False
)

# Report whether evidence fields are populated, without judging their truth.
text_columns = [
    "representative_sample_texts",
    "supporting_evidence",
    "limitations_or_mixed_content",
    "evidence_sampled_records",
    "evidence_nonempty_texts",
    "evidence_distinct_texts",
]

for col in text_columns:
    if col in audit.columns:
        audit[f"has_{col}"] = audit[col].map(nonempty)

# Compare labels across groups. This is a candidate list for human inspection,
# not proof that the topics are equivalent.
pairs = []
records = audit.to_dict("records")

for i in range(len(records)):
    a = records[i]
    la = a["label_normalized"]
    if not la:
        continue

    for j in range(i + 1, len(records)):
        b = records[j]
        lb = b["label_normalized"]
        if not lb or la == lb:
            continue

        similarity = SequenceMatcher(None, la, lb).ratio()
        if similarity >= 0.80:
            pairs.append({
                "model_a": a["model"],
                "community_a": a["community"],
                "topic_a": a["topic"],
                "label_a": a["working_topic_label"],
                "model_b": b["model"],
                "community_b": b["community"],
                "topic_b": b["topic"],
                "label_b": b["working_topic_label"],
                "label_similarity": round(similarity, 3),
                "review_status": "candidate_for_human_comparison",
            })

pairs_df = pd.DataFrame(pairs, columns=[
    "model_a", "community_a", "topic_a", "label_a",
    "model_b", "community_b", "topic_b", "label_b",
    "label_similarity", "review_status"
])

# Keep rejected labels visible but explicitly mark them as not accepted.
audit["interpretation_status"] = "provisional_label_requires_evidence_review"
audit.loc[
    audit["is_do_not_label"], "interpretation_status"
] = "do_not_interpret_as_accepted_label"

audit_file = OUT / "phase11b_topic_quality_audit.csv"
pairs_file = OUT / "phase11b_similar_label_pairs.csv"
summary_file = OUT / "phase11b_quality_audit_summary.csv"

audit.to_csv(audit_file, index=False)
pairs_df.to_csv(pairs_file, index=False)

summary = pd.DataFrame([
    {"check": "topic_groups", "value": len(audit)},
    {"check": "missing_labels", "value": int(audit["label_missing"].sum())},
    {"check": "short_or_generic_label_candidates",
     "value": int(audit["label_short_or_generic"].sum())},
    {"check": "groups_with_exact_duplicate_labels",
     "value": int(audit["exact_duplicate_label"].sum())},
    {"check": "near_duplicate_label_pairs_similarity_at_least_0.80",
     "value": len(pairs_df)},
    {"check": "do_not_label_groups", "value": int(audit["is_do_not_label"].sum())},
    {"check": "batch_approved_groups", "value": int(audit["is_batch_approved"].sum())},
])

summary.to_csv(summary_file, index=False)

print("=" * 72)
print("PHASE 11B — TOPIC QUALITY AUDIT")
print("=" * 72)
print(summary.to_string(index=False))

print("\nLABEL STATUS COUNTS:")
print(audit["interpretation_status"].value_counts().to_string())

print("\nSIMILAR LABEL PAIRS:")
if pairs_df.empty:
    print("No label pairs met the 0.80 similarity threshold.")
else:
    print(pairs_df.to_string(index=False))

print("\nOUTPUT FILES:")
print(audit_file.resolve())
print(pairs_file.resolve())
print(summary_file.resolve())

print("\nNo labels, topic assignments, or decisions were changed.")
print("Similarity is a screening heuristic, not semantic validation.")
