
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "results" / "tables" / "phase10r_expanded_evidence"
OUTPUT_DIR = ROOT / "results" / "tables" / "phase10s_label_revision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = INPUT_DIR / "phase10r_group_evidence_summary.csv"
EXAMPLES_FILE = INPUT_DIR / "phase10r_all_scored_examples.csv"
LOW_FILE = INPUT_DIR / "phase10r_low_similarity_examples.csv"

for path in [SUMMARY_FILE, EXAMPLES_FILE, LOW_FILE]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n"
            "Run Phase 10R first and verify the output files."
        )

summary = pd.read_csv(SUMMARY_FILE)
examples = pd.read_csv(EXAMPLES_FILE)
low_examples = pd.read_csv(LOW_FILE)

keys = ["model", "community", "topic"]
for df in [summary, examples, low_examples]:
    for col in keys:
        if col not in df.columns:
            raise ValueError(f"Missing key column {col} in input file.")
        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()

required_summary = [
    "suggested_label",
    "available_texts",
    "mean_label_similarity",
    "median_label_similarity",
    "lower_similarity_share",
]
missing = [c for c in required_summary if c not in summary.columns]
if missing:
    raise ValueError(f"Summary file is missing columns: {missing}")

if "review_text" not in examples.columns:
    raise ValueError("The all-scored examples file needs a review_text column.")

if "label_semantic_similarity" not in examples.columns:
    raise ValueError(
        "The all-scored examples file needs label_semantic_similarity."
    )

# Rank groups for review. This prioritizes inspection; it does not
# automatically classify topics as good or bad.
summary["review_rank"] = (
    summary["lower_similarity_share"].rank(
        ascending=False, method="min"
    ).astype(int)
)

summary["revision_priority"] = "Standard review"

summary.loc[
    (summary["lower_similarity_share"] >= 0.75)
    | (summary["mean_label_similarity"] < 0.10),
    "revision_priority",
] = "High priority: inspect label and examples"

summary.loc[
    (summary["available_texts"] < 10),
    "revision_priority",
] = "Small sample: inspect all examples"

summary = summary.sort_values(
    ["review_rank", "mean_label_similarity"],
    ascending=[True, True],
).reset_index(drop=True)

# Add empty fields for the reviewer's independent decision.
summary["revised_label"] = ""
summary["human_coherence"] = ""
summary["human_confidence"] = ""
summary["human_decision"] = ""
summary["evidence_for_label"] = ""
summary["evidence_against_label"] = ""
summary["reviewer_notes"] = ""

examples["review_text"] = examples["review_text"].fillna("").astype(str)
examples["label_semantic_similarity"] = pd.to_numeric(
    examples["label_semantic_similarity"], errors="coerce"
)

# Rank examples within each group from lowest to highest similarity.
examples = examples.sort_values(
    keys + ["label_semantic_similarity"],
    ascending=[True, True, True, True],
    na_position="first",
    kind="stable",
).copy()

examples["within_group_similarity_rank"] = (
    examples.groupby(keys).cumcount() + 1
)

examples["human_label_fit"] = ""
examples["human_notes"] = ""

# Add a simple instruction column for Excel review.
examples["review_instruction"] = (
    "Read the post; judge whether it supports the proposed topic. "
    "Do not infer author psychology."
)

# Keep all rows, including those with lower similarity, for transparent review.
group_file = OUTPUT_DIR / "phase10s_group_revision_sheet.csv"
examples_file = OUTPUT_DIR / "phase10s_all_examples_review_sheet.csv"
priority_file = OUTPUT_DIR / "phase10s_high_priority_examples.csv"
low_file = OUTPUT_DIR / "phase10s_low_similarity_examples_reference.csv"
audit_file = OUTPUT_DIR / "phase10s_integrity_summary.csv"

summary.to_csv(group_file, index=False, encoding="utf-8-sig")
examples.to_csv(examples_file, index=False, encoding="utf-8-sig")

# Focus sheet: first five lowest-similarity examples in each group.
priority_examples = (
    examples.groupby(keys, group_keys=False)
    .head(5)
    .copy()
)
priority_examples.to_csv(
    priority_file, index=False, encoding="utf-8-sig"
)

low_examples.to_csv(low_file, index=False, encoding="utf-8-sig")

audit = pd.DataFrame([
    {"check": "groups_in_summary", "value": len(summary)},
    {"check": "scored_example_rows", "value": len(examples)},
    {
        "check": "unique_group_keys_in_examples",
        "value": examples[keys].drop_duplicates().shape[0],
    },
    {
        "check": "groups_with_no_examples",
        "value": len(
            summary.merge(
                examples[keys].drop_duplicates(),
                on=keys,
                how="left",
                indicator=True,
            ).query("_merge == 'left_only'")
        ),
    },
    {
        "check": "high_priority_groups",
        "value": int(
            summary["revision_priority"].str.startswith("High priority").sum()
        ),
    },
    {
        "check": "low_similarity_reference_rows",
        "value": len(low_examples),
    },
])
audit.to_csv(audit_file, index=False, encoding="utf-8-sig")

print("=" * 72)
print("PHASE 10S: LABEL REVISION WORKBOOK")
print("=" * 72)
print(f"Topic groups: {len(summary)}")
print(f"Scored examples: {len(examples)}")
print(f"High-priority groups: {audit.loc[audit.check == 'high_priority_groups', 'value'].iloc[0]}")
print("\nFiles created:")
print(group_file)
print(examples_file)
print(priority_file)
print(low_file)
print(audit_file)

print("\nHOW TO USE:")
print("1. Start with phase10s_group_revision_sheet.csv.")
print("2. Review phase10s_high_priority_examples.csv for initial evidence.")
print("3. Use phase10s_all_examples_review_sheet.csv for deeper inspection.")
print("4. Fill human review columns only after inspecting the post text.")
print("5. Keep model assignments and suggested labels unchanged.")
print("\nSimilarity is a review aid, not a correctness probability.")
print("PHASE 10S COMPLETED")
