
from pathlib import Path
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent

INPUT_FILE = (
    ROOT / "results" / "tables" / "phase10o_a_ai_review_draft"
    / "phase10o_a_example_review_draft.csv"
)
OUTPUT_DIR = ROOT / "results" / "tables" / "phase10r_expanded_evidence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
MAX_TEXT_CHARS = 1500

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n"
        "Run phase10o_a_ai_review_draft.py first."
    )

df = pd.read_csv(INPUT_FILE)

required = [
    "model", "community", "topic", "review_text", "suggested_label"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

df["review_text"] = df["review_text"].fillna("").astype(str)
df["suggested_label"] = df["suggested_label"].fillna("").astype(str)

for col in ["model", "community", "topic"]:
    df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()

df["review_text_clean"] = (
    df["review_text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.slice(0, MAX_TEXT_CHARS)
)

df["text_available"] = df["review_text_clean"].str.len() > 0

# The group key preserves each model's own topic numbering.
group_cols = ["model", "community", "topic"]

# Do not silently mix different labels inside the same group.
label_counts = (
    df.groupby(group_cols)["suggested_label"]
    .nunique()
)
if (label_counts > 1).any():
    bad = label_counts[label_counts > 1]
    raise ValueError(
        "Some groups have multiple suggested labels:\n"
        + bad.to_string()
    )

usable = df.loc[df["text_available"]].copy()
if usable.empty:
    raise ValueError("No non-empty review_text values were found.")

print("=" * 72)
print("PHASE 10R: EXPANDED TOPIC-EVIDENCE AUDIT")
print("=" * 72)
print(f"Input rows: {len(df)}")
print(f"Non-empty texts: {len(usable)}")
print(f"Distinct groups: {df[group_cols].drop_duplicates().shape[0]}")
print(f"Loading local embedding model: {MODEL_NAME}")
print("The model may download weights on its first run if not cached.")

model = SentenceTransformer(MODEL_NAME)

texts = usable["review_text_clean"].tolist()
labels = usable["suggested_label"].tolist()

print("Encoding post texts...")
text_embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

print("Encoding proposed labels...")
label_embeddings = model.encode(
    labels,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

# Cosine similarity is a diagnostic of semantic resemblance,
# not proof that a post supports or contradicts a label.
scores = np.sum(text_embeddings * label_embeddings, axis=1)
usable["label_semantic_similarity"] = scores

# Score bands are descriptive review-priority bands, not validated cutoffs.
def score_band(score):
    if score < 0.20:
        return "lower similarity: inspect carefully"
    if score < 0.35:
        return "intermediate similarity: inspect"
    return "higher similarity: still requires inspection"

usable["similarity_review_flag"] = (
    usable["label_semantic_similarity"].apply(score_band)
)

# Add group-level summary statistics.
group_summary = (
    usable.groupby(group_cols, dropna=False)
    .agg(
        suggested_label=("suggested_label", "first"),
        available_texts=("review_text_clean", "size"),
        mean_label_similarity=("label_semantic_similarity", "mean"),
        median_label_similarity=("label_semantic_similarity", "median"),
        min_label_similarity=("label_semantic_similarity", "min"),
        max_label_similarity=("label_semantic_similarity", "max"),
        sd_label_similarity=("label_semantic_similarity", "std"),
        lower_similarity_count=(
            "label_semantic_similarity",
            lambda x: int((x < 0.20).sum()),
        ),
        intermediate_similarity_count=(
            "label_semantic_similarity",
            lambda x: int(((x >= 0.20) & (x < 0.35)).sum()),
        ),
        higher_similarity_count=(
            "label_semantic_similarity",
            lambda x: int((x >= 0.35).sum()),
        ),
    )
    .reset_index()
)

# Count all rows per group, including rows with empty text.
all_counts = (
    df.groupby(group_cols, dropna=False)
    .agg(
        total_sampled_rows=("review_text", "size"),
        empty_text_rows=("text_available", lambda x: int((~x).sum())),
    )
    .reset_index()
)

group_summary = group_summary.merge(
    all_counts, on=group_cols, how="left", validate="one_to_one"
)

group_summary["lower_similarity_share"] = (
    group_summary["lower_similarity_count"]
    / group_summary["available_texts"].replace(0, np.nan)
)

group_summary["evidence_review_priority"] = np.select(
    [
        group_summary["available_texts"] < 5,
        group_summary["lower_similarity_share"] >= 0.50,
        group_summary["mean_label_similarity"] < 0.20,
    ],
    [
        "small sample: inspect every example",
        "many examples have lower label similarity",
        "low average label similarity: inspect label",
    ],
    default="review actual examples before accepting label",
)

# Select the lowest-scoring examples per group for targeted inspection.
low_examples = (
    usable.sort_values(
        group_cols + ["label_semantic_similarity"],
        ascending=[True, True, True, True],
        kind="stable",
    )
    .groupby(group_cols, group_keys=False)
    .head(5)
    .copy()
)

# Preserve a full row-level audit with original columns.
row_output = usable.copy()

summary_path = OUTPUT_DIR / "phase10r_group_evidence_summary.csv"
rows_path = OUTPUT_DIR / "phase10r_all_scored_examples.csv"
low_path = OUTPUT_DIR / "phase10r_low_similarity_examples.csv"
integrity_path = OUTPUT_DIR / "phase10r_integrity_summary.csv"

group_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
row_output.to_csv(rows_path, index=False, encoding="utf-8-sig")
low_examples.to_csv(low_path, index=False, encoding="utf-8-sig")

integrity = pd.DataFrame([
    {"check": "input_rows", "value": len(df)},
    {"check": "nonempty_text_rows", "value": len(usable)},
    {"check": "empty_text_rows", "value": int((~df["text_available"]).sum())},
    {"check": "distinct_groups", "value": df[group_cols].drop_duplicates().shape[0]},
    {"check": "groups_with_scored_examples", "value": len(group_summary)},
    {"check": "groups_without_scored_examples",
     "value": df[group_cols].drop_duplicates().shape[0] - len(group_summary)},
    {"check": "embedding_model", "value": MODEL_NAME},
    {"check": "similarity_thresholds",
     "value": "0.20 and 0.35 are exploratory review bands, not validated cutoffs"},
])
integrity.to_csv(integrity_path, index=False, encoding="utf-8-sig")

print("\n" + "=" * 72)
print("PHASE 10R RESULTS")
print("=" * 72)
print(f"Rows scored: {len(usable)}")
print(f"Groups with scored examples: {len(group_summary)}")
print(
    "Overall mean label similarity: "
    f"{usable['label_semantic_similarity'].mean():.4f}"
)
print(
    "Overall median label similarity: "
    f"{usable['label_semantic_similarity'].median():.4f}"
)
print("\nFiles saved:")
print(summary_path)
print(rows_path)
print(low_path)
print(integrity_path)

print("\nINTERPRETATION LIMITS:")
print("- Similarity is not a probability that a label is correct.")
print("- Low similarity is not proof of contradiction.")
print("- High similarity is not proof of topic coherence.")
print("- Review the actual text before retaining or rejecting a label.")
print("- No model assignments or source texts were modified.")
print("\nPHASE 10R COMPLETED")
