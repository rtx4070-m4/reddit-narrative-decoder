from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "current_corpus": ROOT / "data" / "processed" / "reddit_nlp_corpus.csv",
    "phase7b_reference": ROOT / "results" / "tables" / "nmf_document_topics_k8_v2.csv",
    "phase8_merged": ROOT / "results" / "tables" / "phase8_cross_community" / "phase8_merged_observations.csv",
}
OUT = ROOT / "results" / "tables" / "phase8_cross_community"
OUT.mkdir(parents=True, exist_ok=True)

def load(name, path):
    if not path.exists():
        print(f"[MISSING] {name}: {path}")
        return None
    df = pd.read_csv(path, low_memory=False)
    if "record_id" not in df.columns:
        print(f"[SKIP] {name}: no record_id column; columns={list(df.columns)}")
        return None
    df = df.copy()
    df["record_id"] = df["record_id"].astype("string").str.strip()
    return df

frames = {k: load(k, v) for k, v in FILES.items()}
frames = {k:v for k,v in frames.items() if v is not None}
if "phase7b_reference" not in frames or "phase8_merged" not in frames:
    raise SystemExit("Need both phase7b_reference and phase8_merged to run the main ID audit.")

sets = {k:set(v["record_id"].dropna().astype(str)) for k,v in frames.items()}
ref = frames["phase7b_reference"]
p8 = frames["phase8_merged"]
ref_ids, p8_ids = sets["phase7b_reference"], sets["phase8_merged"]
extra_ids = sorted(p8_ids - ref_ids)
missing_p8_ids = sorted(ref_ids - p8_ids)

print("=== Phase 7B vs Phase 8 root-cause audit ===")
for k, df in frames.items():
    print(f"{k}: rows={len(df):,}; unique IDs={df['record_id'].nunique(dropna=True):,}")
print(f"Phase 8 IDs absent from Phase 7B: {len(extra_ids):,}")
print(f"Phase 7B IDs absent from Phase 8: {len(missing_p8_ids):,}")

# Enrich the records absent from the Phase 7B reference with both Phase 8 and corpus metadata.
extra = p8[p8["record_id"].astype(str).isin(extra_ids)].copy()
if "current_corpus" in frames:
    corpus = frames["current_corpus"].copy()
    keep = [c for c in [
        "record_id","id","community","subreddit","created_utc","date",
        "text","analysis_text","source_file","analysis_eligible","corpus_version"
    ] if c in corpus.columns]
    corpus = corpus[keep].drop_duplicates("record_id", keep="first")
    # Rename overlapping non-ID fields to retain both versions side-by-side.
    rename = {c:f"corpus_{c}" for c in corpus.columns if c != "record_id" and c in extra.columns}
    corpus = corpus.rename(columns=rename)
    extra = extra.merge(corpus, on="record_id", how="left", validate="one_to_one")

text_candidates = [c for c in ["analysis_text","text","corpus_analysis_text","corpus_text"] if c in extra.columns]
for c in text_candidates:
    extra[c + "_char_len"] = extra[c].fillna("").astype(str).str.len()
    extra[c + "_whitespace_tokens"] = extra[c].fillna("").astype(str).str.split().str.len()

extra.to_csv(OUT / "phase8_extra_ids_root_cause_review_v2.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"record_id":extra_ids}).to_csv(OUT / "phase8_extra_ids_v2.csv", index=False)
pd.DataFrame({"record_id":missing_p8_ids}).to_csv(OUT / "phase7b_ids_missing_from_phase8_v2.csv", index=False)

# Compare reference/corpus ID coverage and basic eligibility flags where possible.
summary = []
for name, df in frames.items():
    row = {"dataset":name,"rows":len(df),"unique_record_ids":df["record_id"].nunique(dropna=True)}
    for col in ["analysis_eligible","corpus_version","community","source_file"]:
        if col in df.columns:
            row[f"{col}_nonmissing"] = int(df[col].notna().sum())
            if col == "analysis_eligible":
                row["analysis_eligible_counts"] = df[col].astype(str).value_counts(dropna=False).to_dict().__str__()
            elif col == "corpus_version":
                row["corpus_version_counts"] = df[col].astype(str).value_counts(dropna=False).to_dict().__str__()
    summary.append(row)
pd.DataFrame(summary).to_csv(OUT / "phase7b_phase8_root_cause_summary_v2.csv", index=False)

print("\nCreated:")
print(OUT / "phase8_extra_ids_root_cause_review_v2.csv")
print(OUT / "phase8_extra_ids_v2.csv")
print(OUT / "phase7b_ids_missing_from_phase8_v2.csv")
print(OUT / "phase7b_phase8_root_cause_summary_v2.csv")
print("\nInterpretation:")
print("- If extra IDs exist in current_corpus but not Phase 7B reference, compare corpus_version and the exact Phase 7B run manifest/input timestamp.")
print("- If extra IDs are absent from current_corpus but present in Phase 8, the pipelines used different corpus snapshots or IDs were introduced downstream.")
print("- Do not add these IDs or refit a model until the intended Phase 7B population is confirmed.")
