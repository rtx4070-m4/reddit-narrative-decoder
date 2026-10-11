from pathlib import Path
import pandas as pd

root = Path(".")
p7j = root / "results/tables/phase7j_nmf/phase7j_cleaned_documents_audit.csv"
corpus = root / "data/processed/reddit_nlp_corpus.csv"
script = root / "src/preprocessing/prepare_nlp_corpus.py"
out = root / "results/tables/phase12a_statistical_audit"
out.mkdir(parents=True, exist_ok=True)

print("=== CORPUS FILES ===")
for path in [p7j, corpus]:
    print(f"{path}: exists={path.exists()}")
    if path.exists():
        print(f"  size={path.stat().st_size:,} bytes")

if p7j.exists() and corpus.exists():
    a = pd.read_csv(p7j, low_memory=False)
    b = pd.read_csv(corpus, low_memory=False)

    print("\n=== COLUMNS ===")
    print("Phase 7J:", list(a.columns))
    print("Processed corpus:", list(b.columns))

    for label, df in [("Phase 7J", a), ("Processed corpus", b)]:
        if "record_id" not in df.columns:
            print(f"{label}: record_id missing")
            continue
        ids = df["record_id"].dropna().astype(str).str.strip()
        print(f"{label}: rows={len(df)}, unique IDs={ids.nunique()}")

    if "record_id" in a.columns and "record_id" in b.columns:
        ids_a = set(a["record_id"].dropna().astype(str).str.strip())
        ids_b = set(b["record_id"].dropna().astype(str).str.strip())
        extra_ids = ids_b - ids_a
        print("\n=== ID COMPARISON ===")
        print("Shared IDs:", len(ids_a & ids_b))
        print("Only in Phase 7J:", len(ids_a - ids_b))
        print("Only in processed corpus:", len(extra_ids))

        extra = b[b["record_id"].astype(str).str.strip().isin(extra_ids)].copy()
        safe_cols = [
            c for c in [
                "record_id", "community", "subreddit",
                "created_utc", "date", "year", "month"
            ] if c in extra.columns
        ]
        extra[safe_cols].to_csv(
            out / "phase12j_processed_corpus_extra_ids.csv", index=False
        )
        print("Saved extra-ID audit:",
              out / "phase12j_processed_corpus_extra_ids.csv")

        for col in ["community", "subreddit"]:
            if col in extra.columns:
                print(f"\nExtra records by {col}:")
                print(extra[col].fillna("MISSING").value_counts().to_string())

        date_col = next(
            (c for c in ["created_utc", "date"] if c in extra.columns), None
        )
        if date_col:
            dates = pd.to_datetime(extra[date_col], errors="coerce", utc=True)
            print("\nExtra records by month:")
            print(
                dates.dt.strftime("%Y-%m").fillna("MISSING")
                .value_counts().sort_index().to_string()
            )

print("\n=== CORPUS PREPARATION SCRIPT ===")
if script.exists():
    lines = script.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    for i, line in enumerate(lines[:220], start=1):
        print(f"{i:4}: {line}")
else:
    print("Preparation script not found:", script)
