from pathlib import Path
import pandas as pd

root = Path(".")
tables = root / "results" / "tables"

paths = {
    "Master corpus": root / "data/processed/reddit_master_clean.csv",
    "Processed NLP corpus": root / "data/processed/reddit_nlp_corpus.csv",
    "Phase 7G audit": tables / "phase7g_nmf/phase7g_cleaned_documents_audit.csv",
    "Phase 7H audit": tables / "phase7h_nmf/phase7h_cleaned_documents_audit.csv",
    "Phase 7I audit": tables / "phase7i_nmf/phase7i_cleaned_documents_audit.csv",
}

frames = {}
id_sets = {}

for name, path in paths.items():
    print(f"\n=== {name} ===")
    print("Path:", path)

    if not path.exists():
        print("MISSING")
        continue

    df = pd.read_csv(path, low_memory=False)
    frames[name] = df

    if "record_id" not in df.columns:
        print("No record_id column. Available columns:", list(df.columns))
        continue

    ids = df["record_id"].dropna().astype(str).str.strip()
    id_sets[name] = set(ids)

    print("Rows:", len(df))
    print("Unique non-null IDs:", ids.nunique())
    print("Duplicate ID rows:", int(ids.duplicated().sum()))
    print("Columns:", list(df.columns))

print("\n=== PAIRWISE ID COMPARISON ===")
names = list(id_sets)

for i, a in enumerate(names):
    for b in names[i + 1:]:
        shared = id_sets[a] & id_sets[b]
        only_a = id_sets[a] - id_sets[b]
        only_b = id_sets[b] - id_sets[a]

        print(f"\n{a} vs {b}")
        print("Shared:", len(shared))
        print(f"Only in {a}:", len(only_a))
        print(f"Only in {b}:", len(only_b))

print("\n=== PHASE 7H INPUT-COLUMN CHECK ===")
p7h = frames.get("Phase 7H audit")
if p7h is not None:
    for col in [
        "topic7g_eligible",
        "topic7g_token_count",
        "topic7h_eligible",
        "topic7h_token_count",
        "source_autotldr_marker",
    ]:
        if col in p7h.columns:
            print(f"\n{col}:")
            print(p7h[col].value_counts(dropna=False).to_string())

print("\n=== SAVE MISSING-ID LISTS ===")
out = tables / "phase12a_statistical_audit"
out.mkdir(parents=True, exist_ok=True)

if "Processed NLP corpus" in id_sets and "Phase 7H audit" in id_sets:
    nlp_only = sorted(
        id_sets["Processed NLP corpus"] - id_sets["Phase 7H audit"]
    )
    nlp = frames["Processed NLP corpus"].copy()
    nlp["record_id"] = nlp["record_id"].astype(str).str.strip()
    missing_rows = nlp[nlp["record_id"].isin(nlp_only)].copy()

    output = out / "phase12n_nlp_ids_absent_from_phase7h.csv"
    missing_rows.to_csv(output, index=False)
    print("NLP IDs absent from Phase 7H:", len(nlp_only))
    print("Saved:", output)

if "Phase 7G audit" in id_sets and "Phase 7H audit" in id_sets:
    p7g_only = sorted(
        id_sets["Phase 7G audit"] - id_sets["Phase 7H audit"]
    )
    print("Phase 7G IDs absent from Phase 7H:", len(p7g_only))

print("\nAudit finished. No source datasets were modified.")
