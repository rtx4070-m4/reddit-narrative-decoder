from pathlib import Path
import pandas as pd

root = Path(".")
tables = root / "results/tables"
out = tables / "phase12a_statistical_audit"
out.mkdir(parents=True, exist_ok=True)

paths = {
    "Master corpus": root / "data/processed/reddit_master_clean.csv",
    "Processed NLP corpus": root / "data/processed/reddit_nlp_corpus.csv",
    "Phase 7J": tables / "phase7j_nmf/phase7j_cleaned_documents_audit.csv",
}

frames = {}
id_sets = {}

for name, path in paths.items():
    print(f"\n=== {name} ===")
    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    df = pd.read_csv(path, low_memory=False)
    frames[name] = df

    if "record_id" not in df.columns:
        print("No record_id column. Available columns:", list(df.columns))
        continue

    ids = df["record_id"].dropna().astype(str).str.strip()
    id_sets[name] = set(ids)

    print("Rows:", len(df))
    print("Nonmissing IDs:", len(ids))
    print("Unique IDs:", ids.nunique())
    print("Columns:", list(df.columns))

print("\n=== RECORD-ID SET COMPARISON ===")
pairs = [
    ("Master corpus", "Processed NLP corpus"),
    ("Master corpus", "Phase 7J"),
    ("Processed NLP corpus", "Phase 7J"),
]

for a, b in pairs:
    if a not in id_sets or b not in id_sets:
        continue
    print(f"\n{a} vs {b}")
    print("Shared IDs:", len(id_sets[a] & id_sets[b]))
    print(f"Only in {a}:", len(id_sets[a] - id_sets[b]))
    print(f"Only in {b}:", len(id_sets[b] - id_sets[a]))

if "Master corpus" in frames and "Processed NLP corpus" in frames:
    master = frames["Master corpus"].copy()
    nlp = frames["Processed NLP corpus"].copy()
    p7j = frames.get("Phase 7J")

    for df in [master, nlp]:
        df["record_id"] = df["record_id"].astype(str).str.strip()

    if p7j is not None:
        p7j = p7j.copy()
        p7j["record_id"] = p7j["record_id"].astype(str).str.strip()

    master_ids = set(master["record_id"])
    nlp_ids = set(nlp["record_id"])
    extra_ids = nlp_ids - id_sets.get("Phase 7J", set())

    extra = nlp[nlp["record_id"].isin(extra_ids)].copy()

    print("\n=== THE 848 RECORDS: MASTER-CORPUS PROFILE ===")
    print("Extra IDs in processed corpus absent from Phase 7J:", len(extra_ids))

    for col in ["community", "subreddit"]:
        if col in extra.columns:
            print(f"\nBy {col}:")
            print(extra[col].fillna("MISSING").value_counts().to_string())

    date_col = next(
        (c for c in ["created_utc", "date"] if c in extra.columns), None
    )
    if date_col:
        dates = pd.to_datetime(extra[date_col], errors="coerce", utc=True)
        print("\nBy month:")
        print(
            dates.dt.strftime("%Y-%m").fillna("MISSING")
            .value_counts().sort_index().to_string()
        )

    if p7j is not None:
        flags = [
            c for c in [
                "topic7j_eligible", "topic7i_eligible",
                "topic7j_residual_template_hint",
                "topic7j_url_artifact_hint",
                "topic7i_source_autotldr_excluded_from_fit",
                "topic7i_residual_template_hint",
                "topic7i_flattened_url_hint",
                "topic_clean_token_count",
            ] if c in p7j.columns
        ]

        matched = extra[["record_id"]].merge(
            p7j[["record_id"] + flags],
            on="record_id",
            how="left",
            validate="one_to_one",
        ) if flags else extra[["record_id"]].copy()

        print("\nPhase 7J cleaning flags on these extra IDs:")
        if flags:
            for col in flags:
                print(f"\n{col}:")
                print(matched[col].value_counts(dropna=False).to_string())
        else:
            print("No expected Phase 7J flags were available for matching.")

    safe_cols = [
        c for c in [
            "record_id", "community", "subreddit",
            "created_utc", "date", "year", "month",
            "is_very_short", "is_long_text", "is_very_long_text",
            "is_article_like", "is_repeated_text", "analysis_eligible",
            "corpus_version",
        ] if c in extra.columns
    ]
    extra[safe_cols].to_csv(
        out / "phase12k_extra_records_master_profile.csv", index=False
    )
    print("\nSaved profile:", out / "phase12k_extra_records_master_profile.csv")
