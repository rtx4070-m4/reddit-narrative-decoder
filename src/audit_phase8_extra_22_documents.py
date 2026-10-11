from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "tables" / "phase8_cross_community"
INPUT = RESULT_DIR / "phase8_ids_absent_from_phase7b_reference.csv"
OUTPUT = RESULT_DIR / "phase8_extra_22_document_review.csv"

def read_csv(path):
    if not path.exists():
        print(f"[MISSING] {path}")
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[ERROR] Could not read {path}: {exc}")
        return None

def find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None

def normalize_id(series):
    return series.astype("string").str.strip()

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Expected audit output not found: {INPUT}\n"
            "Run audit_phase7e_exclusion_reconciliation.py first."
        )

    extra = read_csv(INPUT)
    if extra is None or extra.empty:
        print("No extra documents found in the Phase 8 vs Phase 7B comparison.")
        return

    print("=== Phase 8 documents absent from Phase 7B reference ===")
    print(f"Input rows: {len(extra):,}")
    print("Columns:", list(extra.columns))

    id_col = find_col(extra, ["record_id", "id", "document_id"])
    if id_col is None:
        raise ValueError("Could not identify a record ID column in the input file.")

    # Enrich from the Phase 8 merged observations and NLP corpus, if present.
    enrich_paths = [
        RESULT_DIR / "phase8_merged_observations.csv",
        ROOT / "data" / "processed" / "reddit_nlp_corpus.csv",
        ROOT / "results" / "tables" / "phase7e_nmf" / "phase7e_excluded_short_documents.csv",
        ROOT / "results" / "tables" / "phase7e_nmf" / "phase7e_unmodelled_documents.csv",
    ]

    output = extra.copy()
    output["_audit_id"] = normalize_id(output[id_col])

    for path in enrich_paths:
        df = read_csv(path)
        if df is None or df.empty:
            continue
        src_id = find_col(df, ["record_id", "id", "document_id"])
        if src_id is None:
            print(f"[SKIP] No recognizable ID column in {path.name}")
            continue

        temp = df.copy()
        temp["_audit_id"] = normalize_id(temp[src_id])
        # Avoid duplicate join columns and retain useful metadata/text fields.
        preferred = [
            "community", "subreddit", "date", "created_utc", "year", "month",
            "text", "analysis_text", "topic_clean_text", "word_count",
            "source_file", "source", "meta"
        ]
        chosen = []
        for c in preferred:
            actual = find_col(temp, [c])
            if actual and actual not in chosen and actual != src_id:
                chosen.append(actual)
        if not chosen:
            chosen = [c for c in temp.columns if c not in [src_id, "_audit_id"]][:8]
        temp = temp[["_audit_id"] + chosen].drop_duplicates("_audit_id", keep="first")
        rename = {c: f"{path.stem}__{c}" for c in chosen if c in output.columns}
        temp = temp.rename(columns=rename)
        # Keep first occurrence if the same metadata has already been added.
        output = output.merge(temp, on="_audit_id", how="left", suffixes=("", f"_{path.stem}"))

    # Add convenient normalized audit fields from any available source columns.
    for field in ["community", "subreddit", "date", "text", "analysis_text", "word_count", "source_file"]:
        candidates = [c for c in output.columns if c.lower() == field.lower()]
        if not candidates:
            candidates = [c for c in output.columns if c.lower().endswith("__" + field.lower())]
        if candidates:
            output[f"audit_{field}"] = output[candidates[0]]

    text_col = find_col(output, ["audit_analysis_text", "analysis_text", "text"])
    if text_col:
        output["audit_text_characters"] = output[text_col].fillna("").astype(str).str.len()
        output["audit_text_word_count"] = output[text_col].fillna("").astype(str).str.split().str.len()

    # Print compact review table.
    display_cols = []
    for c in [
        id_col, "audit_community", "audit_subreddit", "audit_date",
        "audit_text_word_count", "audit_text_characters", "audit_source_file",
        "audit_text", "audit_analysis_text"
    ]:
        if c in output.columns and c not in display_cols:
            display_cols.append(c)

    print("\nReview table (text clipped to 220 characters):")
    if display_cols:
        view = output[display_cols].copy()
        for c in ["audit_text", "audit_analysis_text"]:
            if c in view.columns:
                view[c] = view[c].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 220)
        with pd.option_context("display.max_rows", 30, "display.max_columns", 20, "display.width", 220):
            print(view.to_string(index=False))
    else:
        print(output.head(30).to_string(index=False))

    output.drop(columns=["_audit_id"], errors="ignore").to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved enriched review file:\n{OUTPUT}")
    print("\nInterpretation guide:")
    print("- This script describes the 22 records; it does not automatically decide why they were absent.")
    print("- Compare their source IDs, dates, community labels, and text against Phase 7B's input/filtering history.")
    print("- Do not re-fit or modify Phase 7E until the cause is identified.")

if __name__ == "__main__":
    main()
