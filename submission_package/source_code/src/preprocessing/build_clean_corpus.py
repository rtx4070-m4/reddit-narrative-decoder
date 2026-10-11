from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "reddit_master.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "results" / "tables"

CLEAN_FILE = OUTPUT_DIR / "reddit_master_clean.csv"
REPORT_FILE = REPORT_DIR / "corpus_cleaning_report.csv"
CONFLICT_FILE = REPORT_DIR / "source_id_conflicts.csv"


# ============================================================
# SETTINGS
# ============================================================

MIN_TEXT_CHARS = 3


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("BUILDING CLEAN REDDIT ANALYTICAL CORPUS")
print("=" * 70)

print(f"\nInput: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Initial rows: {len(df):,}")


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

df["text"] = df["text"].fillna("").astype(str)

before_text_filter = len(df)

df = df[df["text"].str.strip() != ""].copy()

removed_empty = before_text_filter - len(df)

print(f"Removed empty text: {removed_empty:,}")


# Remove extremely short unusable observations
before_short_filter = len(df)

df = df[df["text"].str.len() >= MIN_TEXT_CHARS].copy()

removed_short = before_short_filter - len(df)

print(f"Removed text < {MIN_TEXT_CHARS} characters: {removed_short:,}")


# ============================================================
# SOURCE-ID AUDIT
# ============================================================

print("\nRunning source-ID audit...")

grouped = (
    df.groupby(["community", "id"], dropna=False)
    .agg(
        records=("id", "size"),
        unique_text=("text", "nunique"),
        unique_time=("created_utc", "nunique"),
        unique_meta=("meta", "nunique"),
        unique_ups=("ups", "nunique"),
        unique_downs=("downs", "nunique"),
    )
    .reset_index()
)


# ============================================================
# CLASSIFY SOURCE-ID GROUPS
# ============================================================

exact_snapshot_groups = grouped[
    (grouped["records"] > 1)
    & (grouped["unique_text"] == 1)
    & (grouped["unique_time"] == 1)
    & (grouped["unique_meta"] == 1)
    & (grouped["unique_ups"] == 1)
    & (grouped["unique_downs"] == 1)
].copy()


vote_only_variation_groups = grouped[
    (grouped["records"] > 1)
    & (grouped["unique_text"] == 1)
    & (grouped["unique_time"] == 1)
    & (grouped["unique_meta"] == 1)
    & (
        (grouped["unique_ups"] > 1)
        | (grouped["unique_downs"] > 1)
    )
].copy()


text_conflict_groups = grouped[
    (grouped["records"] > 1)
    & (grouped["unique_text"] > 1)
].copy()


time_conflict_groups = grouped[
    (grouped["records"] > 1)
    & (grouped["unique_time"] > 1)
].copy()


print(f"\nSource-ID groups: {len(grouped):,}")
print(f"Exact duplicate groups: {len(exact_snapshot_groups):,}")
print(f"Vote-only variation groups: {len(vote_only_variation_groups):,}")
print(f"Text-conflict groups: {len(text_conflict_groups):,}")
print(f"Timestamp-conflict groups: {len(time_conflict_groups):,}")


# ============================================================
# SAVE CONFLICT REPORT
# ============================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

conflict_ids = set(
    zip(
        text_conflict_groups["community"],
        text_conflict_groups["id"],
    )
)

conflict_mask = df.apply(
    lambda row: (row["community"], row["id"]) in conflict_ids,
    axis=1,
)

conflict_rows = df[conflict_mask].copy()

conflict_rows.to_csv(CONFLICT_FILE, index=False)

print(f"\nConflict rows saved to:")
print(CONFLICT_FILE)


# ============================================================
# CONSERVATIVE CLEANING
# ============================================================
#
# We keep one representative observation for each
# community + Reddit ID ONLY when that ID has a single
# unique text.
#
# IDs with conflicting text are excluded from the clean
# analytical corpus and saved separately for review.
# ============================================================

# ============================================================
# RESOLVE TEXT CONFLICTS
# ============================================================
#
# For a source ID with multiple text versions:
# - If one text is a longer version of another, retain
#   the longest available text.
# - Otherwise, keep the ID in the conflict report and
#   exclude it from the analytical corpus.
# ============================================================

resolved_conflict_ids = []

for (community, reddit_id), group in df.groupby(
    ["community", "id"],
    dropna=False
):

    if group["text"].nunique() <= 1:
        continue

    texts = group["text"].dropna().astype(str).unique()

    # Check whether the texts are nested versions of one another.
    # A shorter text must occur within the longest text.
    longest_text = max(texts, key=len)

    nested_versions = all(
        text in longest_text
        for text in texts
    )

    if nested_versions:
        resolved_conflict_ids.append(
            (community, reddit_id)
        )


# IDs whose text conflict can be resolved by retaining
# the longest textual version.
resolved_mask = df.apply(
    lambda row: (row["community"], row["id"])
    in set(resolved_conflict_ids),
    axis=1,
)

# All unresolved conflicts remain excluded.
unresolved_conflict_mask = conflict_mask & ~resolved_mask

# Start with all non-conflicting observations.
clean_candidates = df[~unresolved_conflict_mask].copy()

# For resolved conflicts, retain the longest text.
resolved_rows = df[resolved_mask].copy()

if len(resolved_rows) > 0:

    resolved_rows["_text_length"] = (
        resolved_rows["text"].str.len()
    )

    resolved_rows = (
        resolved_rows
        .sort_values(
            ["community", "id", "_text_length"],
            ascending=[True, True, False]
        )
        .drop_duplicates(
            subset=["community", "id"],
            keep="first"
        )
        .drop(columns=["_text_length"])
    )

    # Remove all original rows belonging to resolved IDs.
    clean_candidates = clean_candidates[
        ~clean_candidates.apply(
            lambda row: (
                row["community"],
                row["id"]
            ) in set(resolved_conflict_ids),
            axis=1
        )
    ]

    clean_candidates = pd.concat(
        [clean_candidates, resolved_rows],
        ignore_index=True
    )

# Sort deterministically so that representative selection
# is reproducible.
sort_columns = [
    "community",
    "id",
    "created_utc",
    "record_id",
]

existing_sort_columns = [
    c for c in sort_columns if c in clean_candidates.columns
]

clean_candidates = clean_candidates.sort_values(
    existing_sort_columns
)

# One representative row per community + Reddit ID
clean = (
    clean_candidates
    .drop_duplicates(
        subset=["community", "id"],
        keep="first"
    )
    .copy()
)


# ============================================================
# CREATE FINAL RECORD ID
# ============================================================

clean = clean.reset_index(drop=True)

clean["record_id"] = [
    f"{community}_{i:06d}"
    for i, community in enumerate(
        clean["community"],
        start=1
    )
]


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

preferred_columns = [
    "record_id",
    "id",
    "text",
    "subreddit",
    "community",
    "meta",
    "created_utc",
    "date",
    "year",
    "month",
    "day",
    "hour",
    "ups",
    "downs",
    "source_file",
]

final_columns = [
    c for c in preferred_columns
    if c in clean.columns
]

clean = clean[final_columns]


# ============================================================
# SAVE CLEAN CORPUS
# ============================================================

clean.to_csv(CLEAN_FILE, index=False)


# ============================================================
# CLEANING REPORT
# ============================================================

community_raw = (
    df["community"]
    .value_counts()
    .rename_axis("community")
    .reset_index(name="rows_after_text_filter")
)

community_clean = (
    clean["community"]
    .value_counts()
    .rename_axis("community")
    .reset_index(name="final_unique_observations")
)

community_report = community_raw.merge(
    community_clean,
    on="community",
    how="outer"
).fillna(0)

community_report["rows_removed_by_deduplication"] = (
    community_report["rows_after_text_filter"]
    - community_report["final_unique_observations"]
)

community_report["removal_percent"] = (
    community_report["rows_removed_by_deduplication"]
    / community_report["rows_after_text_filter"]
    * 100
)


# Add overall summary
summary_rows = pd.DataFrame(
    [
        {
            "metric": "initial_master_rows",
            "value": before_text_filter,
        },
        {
            "metric": "removed_empty_text",
            "value": removed_empty,
        },
        {
            "metric": "removed_short_text",
            "value": removed_short,
        },
        {
            "metric": "rows_after_text_filter",
            "value": len(df),
        },
        {
            "metric": "source_id_groups",
            "value": len(grouped),
        },
        {
            "metric": "exact_duplicate_groups",
            "value": len(exact_snapshot_groups),
        },
        {
            "metric": "vote_only_variation_groups",
            "value": len(vote_only_variation_groups),
        },
        {
            "metric": "text_conflict_groups",
            "value": len(text_conflict_groups),
        },
        {
            "metric": "timestamp_conflict_groups",
            "value": len(time_conflict_groups),
        },
        {
            "metric": "final_unique_observations",
            "value": len(clean),
        },
    ]
)

# Save summary and community report together
report_path = REPORT_FILE

with pd.ExcelWriter(
    REPORT_DIR / "corpus_cleaning_report.xlsx",
    engine="openpyxl"
) as writer:

    summary_rows.to_excel(
        writer,
        sheet_name="summary",
        index=False
    )

    community_report.to_excel(
        writer,
        sheet_name="community",
        index=False
    )

    grouped.to_excel(
        writer,
        sheet_name="source_id_audit",
        index=False
    )


# Also save CSV summary
summary_rows.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("CLEAN CORPUS CREATED")
print("=" * 70)

print(f"\nFinal observations: {len(clean):,}")

print("\nFinal observations by community:")
print(
    clean["community"]
    .value_counts()
    .to_string()
)

print("\nFinal date range:")

dates = pd.to_datetime(
    clean["created_utc"],
    errors="coerce"
)

print("Minimum:", dates.min())
print("Maximum:", dates.max())

print("\nFiles created:")

print(f"Clean corpus:")
print(CLEAN_FILE)

print(f"\nCleaning report:")
print(REPORT_DIR / "corpus_cleaning_report.xlsx")

print(f"\nText-conflict records:")
print(CONFLICT_FILE)

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "\nRows with multiple unique texts for the same "
    "community + Reddit ID were NOT silently discarded."
)

print(
    "They have been isolated in source_id_conflicts.csv "
    "for separate investigation."
)