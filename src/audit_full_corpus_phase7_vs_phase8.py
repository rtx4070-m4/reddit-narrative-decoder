from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "phase8": ROOT / "results" / "tables" / "phase8_cross_community"
              / "phase8_merged_observations.csv",
    "phase7e": ROOT / "results" / "tables" / "phase7e_nmf"
               / "phase7e_document_topics_k8.csv",
    "phase7b_reference": ROOT / "results" / "tables"
                        / "nmf_document_topics_k8_v2.csv",
    "nlp_corpus": ROOT / "data" / "processed" / "reddit_nlp_corpus.csv",
}


def find_col(df, candidates, file_label):
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(
        f"{file_label}: expected one of {candidates}; "
        f"available columns: {list(df.columns)}"
    )


def load_dataset(label, path):
    if not path.exists():
        print(f"\n[{label}] FILE NOT FOUND: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    id_col = find_col(df, ["record_id"], label)
    community_col = find_col(df, ["community", "subreddit"], label)

    date_col = None
    for candidate in ["created_utc", "date", "master_date"]:
        try:
            date_col = find_col(df, [candidate], label)
            break
        except ValueError:
            pass

    df["_audit_id"] = df[id_col].astype("string").str.strip()
    df["_audit_community"] = df[community_col].astype("string").str.strip()

    if date_col is not None:
        df["_audit_date"] = pd.to_datetime(
            df[date_col], errors="coerce", utc=True
        )
        df["_audit_month"] = df["_audit_date"].dt.strftime("%Y-%m")
    else:
        df["_audit_date"] = pd.NaT
        df["_audit_month"] = pd.NA

    print(f"\n[{label}]")
    print(f"Path: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Unique non-empty record IDs: {df['_audit_id'].dropna().nunique():,}")
    print("Rows by community:")
    print(df.groupby("_audit_community", dropna=False).size().to_string())

    if df["_audit_month"].notna().any():
        print("Rows by month:")
        print(df.groupby("_audit_month", dropna=False).size().to_string())

    return df


datasets = {
    label: load_dataset(label, path)
    for label, path in FILES.items()
}

available = {k: v for k, v in datasets.items() if v is not None}
if "phase8" not in available or "phase7e" not in available:
    raise SystemExit(
        "\nRequired Phase 8 or Phase 7E file is missing. "
        "Check the path printed above."
    )

out = ROOT / "results" / "tables" / "phase8_cross_community"
out.mkdir(parents=True, exist_ok=True)

# Compare full record-ID sets, not only February.
base_ids = set(available["phase8"]["_audit_id"].dropna().astype(str))
for label, df in available.items():
    ids = set(df["_audit_id"].dropna().astype(str))
    print(f"\nFULL-CORPUS COMPARISON: Phase 8 vs {label}")
    print(f"Phase 8 unique IDs: {len(base_ids):,}")
    print(f"{label} unique IDs: {len(ids):,}")
    print(f"Phase 8 IDs absent from {label}: {len(base_ids - ids):,}")
    print(f"{label} IDs absent from Phase 8: {len(ids - base_ids):,}")

    if label == "phase8":
        continue

    missing = available["phase8"][
        available["phase8"]["_audit_id"].astype(str).isin(base_ids - ids)
    ].copy()
    extra = df[df["_audit_id"].astype(str).isin(ids - base_ids)].copy()

    if "_audit_month" in missing.columns:
        print(f"Phase 8 IDs absent from {label}, by month:")
        print(missing.groupby("_audit_month", dropna=False).size().to_string())
        print(f"Phase 8 IDs absent from {label}, by community:")
        print(
            missing.groupby("_audit_community", dropna=False)
            .size().to_string()
        )

    missing.drop(
        columns=["_audit_id", "_audit_community", "_audit_date", "_audit_month"],
        errors="ignore",
    ).to_csv(out / f"audit_phase8_ids_missing_from_{label}.csv", index=False)

    extra.drop(
        columns=["_audit_id", "_audit_community", "_audit_date", "_audit_month"],
        errors="ignore",
    ).to_csv(out / f"audit_{label}_ids_absent_from_phase8.csv", index=False)

# Direct comparison among Phase 7B reference, Phase 7E, and NLP corpus.
if all(k in available for k in ["phase7b_reference", "phase7e", "nlp_corpus"]):
    print("\nPHASE 7B REFERENCE vs PHASE 7E vs NLP CORPUS")
    for label in ["phase7b_reference", "phase7e", "nlp_corpus"]:
        ids = set(available[label]["_audit_id"].dropna().astype(str))
        print(f"{label}: {len(ids):,} unique IDs")

    ref_ids = set(available["phase7b_reference"]["_audit_id"].dropna().astype(str))
    e_ids = set(available["phase7e"]["_audit_id"].dropna().astype(str))
    nlp_ids = set(available["nlp_corpus"]["_audit_id"].dropna().astype(str))

    print(f"Phase 7B reference IDs absent from Phase 7E: {len(ref_ids - e_ids):,}")
    print(f"Phase 7E IDs absent from Phase 7B reference: {len(e_ids - ref_ids):,}")
    print(f"NLP corpus IDs absent from Phase 7B reference: {len(nlp_ids - ref_ids):,}")
    print(f"Phase 7B reference IDs absent from NLP corpus: {len(ref_ids - nlp_ids):,}")

print(f"\nAudit outputs saved to: {out}")
