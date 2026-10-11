from pathlib import Path
import pandas as pd

root = Path(".")
script = root / "src/topics/nmf_topic_model_v5.py"
lines = script.read_text(encoding="utf-8-sig", errors="replace").splitlines()

print("=== PHASE 7E SOURCE LOGIC ===")
terms = [
    "MASTER_PATH", "PHASE7B_PATH", "OUT =",
    "read_csv", "dropna", "drop_duplicates",
    "merge(", "isin(", "eligible", "MIN_CLEAN_TOKENS",
    "to_csv", "def main", "def load", "def run",
    "original_text", "record_id", "if __name__",
]

shown = set()
for i, line in enumerate(lines):
    if any(term.lower() in line.lower() for term in terms):
        for j in range(max(0, i - 5), min(len(lines), i + 9)):
            if j not in shown:
                print(f"{j + 1:4}: {lines[j]}")
                shown.add(j)

print("\n=== CONFIGURED INPUT FILE COUNTS ===")
paths = [
    root / "data/processed/reddit_master.csv",
    root / "data/processed/reddit_master_clean.csv",
    root / "data/processed/reddit_nlp_corpus.csv",
    root / "results/tables/nmf_document_topics_k8_v2.csv",
    root / "results/tables/phase7e_nmf/phase7e_document_topics_k8.csv",
    root / "results/tables/phase7e_nmf/phase7e_excluded_short_documents.csv",
    root / "results/tables/phase7e_nmf/phase7e_unmodelled_documents.csv",
]

for path in paths:
    print(f"\n{path}")
    if not path.exists():
        print("MISSING")
        continue
    try:
        df = pd.read_csv(path, low_memory=False)
        print("Rows:", len(df))
        if "record_id" in df.columns:
            ids = df["record_id"].dropna().astype(str).str.strip()
            print("Unique record IDs:", ids.nunique())
            print("Duplicate ID rows:", int(ids.duplicated().sum()))
        print("Columns:", list(df.columns))
    except Exception as exc:
        print("READ ERROR:", repr(exc))

print("\nInspection only: no data or model outputs modified.")
