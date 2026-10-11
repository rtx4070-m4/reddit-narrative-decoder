
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "results" / "tables" / "phase10o_a_ai_review_draft"
OUTPUT_DIR = ROOT / "results" / "tables" / "phase10p_review_pack"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES_FILE = INPUT_DIR / "phase10o_a_example_review_draft.csv"
GROUPS_FILE = INPUT_DIR / "phase10o_a_group_review_draft.csv"

if not EXAMPLES_FILE.exists() or not GROUPS_FILE.exists():
    raise FileNotFoundError(
        "Phase 10O-A draft files were not found. Run phase10o_a_ai_review_draft.py first."
    )

examples = pd.read_csv(EXAMPLES_FILE)
groups = pd.read_csv(GROUPS_FILE)

keys = ["model", "community", "topic"]
for df in (examples, groups):
    for col in keys:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")
        df[col] = df[col].astype(str).str.strip()

# Locate the original post-text column without changing the source data.
text_candidates = [
    "text",
    "analysis_text",
    "review_text",
    "body",
    "post_text",
    "content",
]
text_col = next((c for c in text_candidates if c in examples.columns), None)

if text_col is None:
    raise ValueError(
        "Could not identify the post-text column. Available columns: "
        + ", ".join(examples.columns)
    )

examples[text_col] = examples[text_col].fillna("").astype(str)
examples["review_text_excerpt"] = (
    examples[text_col].str.replace(r"\s+", " ", regex=True).str.slice(0, 500)
)

# Group-level evidence summary.
group_counts = (
    examples.groupby(keys, dropna=False)
    .agg(
        sampled_examples=(text_col, "size"),
        nonempty_texts=("review_text_excerpt", lambda s: int((s.str.len() > 0).sum())),
        distinct_texts=("review_text_excerpt", "nunique"),
    )
    .reset_index()
)

group_review = groups.merge(
    group_counts, on=keys, how="left", validate="one_to_one"
)

# Flag groups that deserve additional scrutiny.
def review_flag(row):
    n = row.get("sampled_examples", 0)
    label = str(row.get("suggested_label", ""))
    confidence = str(row.get("suggested_confidence", "")).lower()

    if pd.isna(n) or n < 5:
        return "VERY SMALL SAMPLE: inspect every example"
    if confidence == "low":
        return "LOW CONFIDENCE: inspect for mixed or generic content"
    if "mixed" in str(row.get("suggested_coherence", "")).lower():
        return "MIXED THEME: check whether one label fits most examples"
    if "general" in label.lower() or "mixed" in label.lower():
        return "BROAD LABEL: consider a more specific theme"
    return "REVIEW EXAMPLES BEFORE ACCEPTING LABEL"

group_review["review_priority"] = group_review.apply(review_flag, axis=1)

# Make a compact evidence sheet with up to five actual examples per group.
examples = examples.sort_values(keys, kind="stable")
examples["example_number"] = examples.groupby(keys).cumcount() + 1
evidence = examples.loc[
    examples["example_number"] <= 5
].copy()

wanted = keys + [
    "example_number",
    "review_text_excerpt",
    "suggested_label",
    "suggested_coherence",
    "suggested_confidence",
    "suggested_decision",
]
evidence = evidence[[c for c in wanted if c in evidence.columns]]

# Preserve all original group columns and append evidence counts.
group_path = OUTPUT_DIR / "phase10p_group_evidence_review.csv"
evidence_path = OUTPUT_DIR / "phase10p_first_five_examples.csv"
audit_path = OUTPUT_DIR / "phase10p_integrity_summary.csv"

group_review.to_csv(group_path, index=False, encoding="utf-8-sig")
evidence.to_csv(evidence_path, index=False, encoding="utf-8-sig")

summary = pd.DataFrame([
    {"check": "example_rows", "value": len(examples)},
    {"check": "unique_example_record_keys", "value": examples[keys].drop_duplicates().shape[0]},
    {"check": "group_rows", "value": len(groups)},
    {"check": "unique_group_keys", "value": groups[keys].drop_duplicates().shape[0]},
    {"check": "groups_with_examples", "value": int(group_counts["sampled_examples"].notna().sum())},
    {"check": "groups_without_examples", "value": int(group_counts["sampled_examples"].isna().sum())},
    {"check": "identified_text_column", "value": text_col},
    {"check": "examples_with_empty_text", "value": int((examples[text_col].str.strip() == "").sum())},
    {"check": "suggestions_missing", "value": int(examples["suggested_label"].isna().sum())},
])
summary.to_csv(audit_path, index=False, encoding="utf-8-sig")

print("=" * 72)
print("PHASE 10P: TOPIC REVIEW PACK")
print("=" * 72)
print(f"Examples: {len(examples)}")
print(f"Topic groups: {len(groups)}")
print(f"Text column: {text_col}")
print(f"Groups with examples: {group_counts['sampled_examples'].notna().sum()}")
print(f"Groups without examples: {group_counts['sampled_examples'].isna().sum()}")
print(f"\nSaved: {group_path}")
print(f"Saved: {evidence_path}")
print(f"Saved: {audit_path}")
print("\nOpen phase10p_first_five_examples.csv to inspect real post text.")
print("Open phase10p_group_evidence_review.csv to review group suggestions.")
print("These are review aids, not completed human-validation results.")
print("\nPHASE 10P COMPLETED")
