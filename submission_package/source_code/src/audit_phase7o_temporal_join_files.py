from pathlib import Path
import pandas as pd

base = Path("results/tables/phase7o_temporal_narratives_v17")

files = {
    "NMF": base / "nmf_document_topic_date_join.csv",
    "Semantic_KMeans": base / "semantic_kmeans_document_topic_date_join.csv",
}

for model, path in files.items():
    print("\n" + "=" * 70)
    print(model, "TEMPORAL JOIN AUDIT")
    print("=" * 70)

    if not path.exists():
        print("MISSING FILE:", path)
        continue

    df = pd.read_csv(path)
    print("File:", path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Identify likely record-ID column.
    id_candidates = [
        c for c in df.columns
        if c.lower() in {"record_id", "id", "document_id", "post_id"}
    ]
    if not id_candidates:
        id_candidates = [c for c in df.columns if "id" in c.lower()]

    if id_candidates:
        id_col = id_candidates[0]
        print("ID column:", id_col)
        print("Unique IDs:", df[id_col].nunique(dropna=True))
        print("Missing IDs:", int(df[id_col].isna().sum()))
        print("Duplicate ID rows:", int(df[id_col].duplicated(keep=False).sum()))
    else:
        print("No obvious ID column detected.")

    date_candidates = [
        c for c in df.columns
        if any(term in c.lower() for term in ["date", "created", "month", "time"])
    ]
    print("Potential date columns:", date_candidates)

    for col in date_candidates:
        if "month" in col.lower():
            print(f"{col} distinct values:", df[col].nunique(dropna=True))
            print(df[col].value_counts(dropna=False).head(12).to_string())
        elif "date" in col.lower() or "created" in col.lower() or "time" in col.lower():
            dates = pd.to_datetime(df[col], errors="coerce", utc=True)
            print(
                f"{col}: valid={int(dates.notna().sum())}, "
                f"missing_or_invalid={int(dates.isna().sum())}"
            )
            if dates.notna().any():
                print("Date range:", dates.min(), "to", dates.max())

    topic_candidates = [
        c for c in df.columns
        if "topic" in c.lower() or "assigned" in c.lower()
    ]
    print("Potential topic columns:", topic_candidates)

    for col in topic_candidates:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False).head(15).to_string())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

