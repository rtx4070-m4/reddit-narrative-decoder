from pathlib import Path
import pandas as pd

root = Path(".")
tables = root / "results/tables"

paths = {
    "Master corpus": root / "data/processed/reddit_master_clean.csv",
    "Processed NLP corpus": root / "data/processed/reddit_nlp_corpus.csv",
    "Phase 7I input": tables / "phase7i_nmf/phase7i_cleaned_documents_audit.csv",
    "Phase 7J output": tables / "phase7j_nmf/phase7j_cleaned_documents_audit.csv",
}

frames = {}
sets = {}

for name, path in paths.items():
    print(f"\n=== {name} ===")
    if not path.exists():
        print("MISSING:", path)
        continue

    df = pd.read_csv(path, low_memory=False)
    frames[name] = df

    if "record_id" not in df.columns:
        print("No record_id column. Columns:", list(df.columns))
        continue

    ids = df["record_id"].dropna().astype(str).str.strip()
    sets[name] = set(ids)
    print(f"Rows={len(df)}, unique IDs={ids.nunique()}")
    print("Columns:", list(df.columns))

print("\n=== ID-SET DIFFERENCES ===")
names = list(sets)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        print(f"\n{a} vs {b}")
        print("Shared:", len(sets[a] & sets[b]))
        print(f"Only in {a}:", len(sets[a] - sets[b]))
        print(f"Only in {b}:", len(sets[b] - sets[a]))

# Inspect Phase 7I eligibility flags for the 848 known extra NLP IDs.
if "Processed NLP corpus" in sets and "Phase 7I input" in sets:
    extra_ids = sets["Processed NLP corpus"] - sets.get("Phase 7J output", set())
    p7i = frames["Phase 7I input"].copy()
    p7i["record_id"] = p7i["record_id"].astype(str).str.strip()
    extra_rows = p7i[p7i["record_id"].isin(extra_ids)].copy()

    print("\n=== EXTRA NLP IDS PRESENT IN PHASE 7I ===")
    print("Extra NLP IDs:", len(extra_ids))
    print("Found in Phase 7I:", len(extra_rows))
    print("Not found in Phase 7I:", len(extra_ids - sets["Phase 7I input"]))

    flags = [
        c for c in p7i.columns
        if any(term in c.lower() for term in [
            "eligible", "token_count", "excluded", "artifact", "template"
        ])
    ]
    for col in flags:
        print(f"\n{col}:")
        print(extra_rows[col].value_counts(dropna=False).to_string())

    out = tables / "phase12a_statistical_audit"
    out.mkdir(parents=True, exist_ok=True)
    extra_rows.to_csv(out / "phase12m_phase7i_extra_record_flags.csv", index=False)
    print("\nSaved:", out / "phase12m_phase7i_extra_record_flags.csv")
