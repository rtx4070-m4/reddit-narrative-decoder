from pathlib import Path
import pandas as pd

files = [
    "results/tables/phase10o_human_validation/phase10o_example_rating_sheet.csv",
    "results/tables/phase10o_human_validation/phase10o_group_decision_sheet.csv",
    "results/tables/phase10p_review_pack/phase10p_group_evidence_review.csv",
    "results/tables/phase10r_expanded_evidence/phase10r_group_evidence_summary.csv",
    "results/tables/phase10r_expanded_evidence/phase10r_integrity_summary.csv",
    "results/tables/phase10s_label_revision/phase10s_group_revision_sheet.csv",
    "results/tables/phase10s_label_revision/phase10s_integrity_summary.csv",
]

for filename in files:
    path = Path(filename)
    print("\n" + "=" * 78)
    print(filename)
    print("=" * 78)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Report non-empty values in likely review and decision fields.
    review_cols = [
        c for c in df.columns
        if any(term in c.lower() for term in [
            "label", "rating", "decision", "review", "note",
            "revision", "status", "confidence", "interpret"
        ])
    ]

    for col in review_cols:
        values = df[col].fillna("").astype(str).str.strip()
        filled = values.ne("")
        print(f"{col}: {int(filled.sum())}/{len(df)} populated")
        if filled.any() and values.nunique() <= 15:
            print("  Values:", values[filled].value_counts().to_dict())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False, max_cols=20, max_colwidth=100))
