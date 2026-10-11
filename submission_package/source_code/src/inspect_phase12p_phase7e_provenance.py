from pathlib import Path
import pandas as pd

root = Path(".")
terms = [
    "phase7e_document_topics_k8.csv",
    "phase7e_nmf",
]

print("=== SOURCE SCRIPT SEARCH ===")
for path in sorted((root / "src").rglob("*.py")):
    if path.name.startswith(("inspect_phase12", "run_phase12")):
        continue
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        continue

    hits = [i for i, line in enumerate(lines)
            if any(term.lower() in line.lower() for term in terms)]

    if hits:
        print(f"\n--- {path} ---")
        shown = set()
        for i in hits:
            for j in range(max(0, i - 8), min(len(lines), i + 12)):
                if j not in shown:
                    print(f"{j + 1:4}: {lines[j]}")
                    shown.add(j)

print("\n=== PHASE 7E FILE AND CORPUS COUNTS ===")
paths = {
    "Master corpus": root / "data/processed/reddit_master_clean.csv",
    "Processed NLP corpus": root / "data/processed/reddit_nlp_corpus.csv",
    "Phase 7E document topics": root / "results/tables/phase7e_nmf/phase7e_document_topics_k8.csv",
    "Phase 7G audit": root / "results/tables/phase7g_nmf/phase7g_cleaned_documents_audit.csv",
}

sets = {}
for name, path in paths.items():
    print(f"\n{name}: {path}")
    if not path.exists():
        print("MISSING")
        continue

    df = pd.read_csv(path, low_memory=False)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    if "record_id" in df.columns:
        ids = df["record_id"].dropna().astype(str).str.strip()
        sets[name] = set(ids)
        print("Unique IDs:", len(sets[name]))
        print("Duplicate ID rows:", int(ids.duplicated().sum()))
    else:
        print("WARNING: record_id column missing")

print("\n=== ID SET DIFFERENCES ===")
for a, b in [
    ("Processed NLP corpus", "Phase 7E document topics"),
    ("Phase 7E document topics", "Phase 7G audit"),
    ("Processed NLP corpus", "Phase 7G audit"),
]:
    if a in sets and b in sets:
        print(f"\n{a} vs {b}")
        print("Shared IDs:", len(sets[a] & sets[b]))
        print(f"Only in {a}:", len(sets[a] - sets[b]))
        print(f"Only in {b}:", len(sets[b] - sets[a]))

print("\nInspection complete; no source data modified.")
