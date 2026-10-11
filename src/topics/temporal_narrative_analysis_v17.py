#!/usr/bin/env python
"""Phase 7O corrected: temporal topic prevalence using explicit model configurations.

Defaults:
  NMF: variant=unigram_bigram, k=5 (Phase 7K)
  Semantic K-means: k=5 (Phase 7L)

Run from project root:
  python src/topics/temporal_narrative_analysis_v17.py
  python src/topics/temporal_narrative_analysis_v17.py --run-model

Override selections if needed:
  --nmf-variant unigram_bigram --nmf-k 5 --semantic-k 5

This script filters configurations BEFORE joining to the corpus, joins on record_id,
checks duplicate keys, and explicitly retains missing assignments as UNASSIGNED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
DEFAULT_CORPUS = TABLES / "phase7j_nmf" / "phase7j_cleaned_documents_audit.csv"
DEFAULT_NMF = TABLES / "phase7k_nmf" / "phase7k_document_topics.csv"
DEFAULT_SEMANTIC = TABLES / "phase7l_semantic" / "phase7l_document_topics.csv"
DEFAULT_OUT = TABLES / "phase7o_temporal_narratives_v17"


def find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def require_file(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path, low_memory=False)


def parse_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        nums = pd.to_numeric(series, errors="coerce")
        median = nums.dropna().median() if nums.notna().any() else np.nan
        unit = "s" if pd.notna(median) and median > 100_000_000 else "ms"
        return pd.to_datetime(nums, unit=unit, errors="coerce", utc=True).dt.tz_convert(None)
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)


def load_corpus(path: Path) -> pd.DataFrame:
    df = require_file(path, "Corpus")
    id_col = find_col(df, ["record_id", "document_id", "doc_id"])
    date_col = find_col(df, ["date", "created_date", "created_utc", "timestamp", "created_at"])
    community_col = find_col(df, ["community", "subreddit", "subreddit_name"])
    if not id_col or not date_col or not community_col:
        raise ValueError(
            "Corpus must have record_id, date, and community/subreddit columns. "
            f"Available columns: {df.columns.tolist()}"
        )
    out = pd.DataFrame({
        "record_id": df[id_col].astype(str),
        "analysis_date": parse_date(df[date_col]),
        "community": df[community_col].astype(str),
    })
    out["month"] = out["analysis_date"].dt.to_period("M").astype(str)
    # Keep source order and make duplicate-key problem visible rather than silently choosing a row.
    duplicate_ids = int(out["record_id"].duplicated(keep=False).sum())
    if duplicate_ids:
        raise ValueError(
            f"Corpus contains {duplicate_ids} rows with duplicate record_id values. "
            "Resolve duplicates before temporal analysis."
        )
    return out


def select_nmf(path: Path, variant: str, k: int) -> pd.DataFrame:
    df = require_file(path, "NMF assignment file")
    required = ["record_id", "variant", "k", "dominant_topic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"NMF file missing required columns {missing}. Found: {df.columns.tolist()}")
    selected = df[
        df["variant"].astype(str).str.strip().str.lower().eq(variant.strip().lower())
        & pd.to_numeric(df["k"], errors="coerce").eq(k)
    ].copy()
    if selected.empty:
        configs = df[["variant", "k"]].drop_duplicates().to_dict("records")
        raise ValueError(f"No NMF rows for variant={variant!r}, k={k}. Available: {configs}")
    if selected["record_id"].astype(str).duplicated().any():
        raise ValueError(f"NMF selection {variant}, k={k} has duplicate record_id values.")
    selected["record_id"] = selected["record_id"].astype(str)
    selected["topic_label"] = "Topic " + pd.to_numeric(
        selected["dominant_topic"], errors="coerce"
    ).astype("Int64").astype(str)
    selected["model_assignment"] = "NMF"
    selected["model_configuration"] = f"{variant}, K={k}"
    keep = ["record_id", "topic_label", "model_assignment", "model_configuration"]
    if "dominant_topic_share" in selected.columns:
        selected["assignment_strength"] = pd.to_numeric(selected["dominant_topic_share"], errors="coerce")
        keep.append("assignment_strength")
    return selected[keep]


def select_semantic(path: Path, k: int) -> pd.DataFrame:
    df = require_file(path, "Semantic assignment file")
    required = ["record_id", "k", "topic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Semantic file missing required columns {missing}. Found: {df.columns.tolist()}")
    selected = df[pd.to_numeric(df["k"], errors="coerce").eq(k)].copy()
    if selected.empty:
        available = sorted(pd.to_numeric(df["k"], errors="coerce").dropna().unique().tolist())
        raise ValueError(f"No semantic rows for k={k}. Available k values: {available}")
    if selected["record_id"].astype(str).duplicated().any():
        raise ValueError(f"Semantic selection K={k} has duplicate record_id values.")
    selected["record_id"] = selected["record_id"].astype(str)
    selected["topic_label"] = "Topic " + pd.to_numeric(
        selected["topic"], errors="coerce"
    ).astype("Int64").astype(str)
    selected["model_assignment"] = "Semantic_KMeans"
    selected["model_configuration"] = f"K={k}"
    keep = ["record_id", "topic_label", "model_assignment", "model_configuration"]
    if "centroid_cosine_distance" in selected.columns:
        selected["assignment_strength"] = pd.to_numeric(
            selected["centroid_cosine_distance"], errors="coerce"
        )
        keep.append("assignment_strength")
    return selected[keep]


def make_summaries(corpus: pd.DataFrame, assignments: pd.DataFrame, model_key: str,
                   outdir: Path) -> dict:
    merged = corpus.merge(assignments, on="record_id", how="left", validate="one_to_one")
    merged["topic_label"] = merged["topic_label"].fillna("UNASSIGNED")
    merged["model_assignment"] = model_key
    merged["model_configuration"] = merged["model_configuration"].fillna("selected configuration")
    merged["has_assignment"] = merged["topic_label"].ne("UNASSIGNED")
    merged["month"] = merged["analysis_date"].dt.to_period("M").astype(str)

    dated = merged[merged["analysis_date"].notna()].copy()
    if dated.empty:
        raise ValueError(f"No dated corpus rows for {model_key}")

    # Main prevalence includes UNASSIGNED in the denominator, so coverage is transparent.
    monthly = dated.groupby(["month", "topic_label"], dropna=False).size().rename("document_count").reset_index()
    totals = dated.groupby("month").size().rename("month_total_all_corpus_rows").reset_index()
    assigned_totals = dated[dated["has_assignment"]].groupby("month").size().rename(
        "month_total_assigned_rows"
    ).reset_index()
    monthly = monthly.merge(totals, on="month", how="left")
    monthly = monthly.merge(assigned_totals, on="month", how="left")
    monthly["month_total_assigned_rows"] = monthly["month_total_assigned_rows"].fillna(0).astype(int)
    monthly["share_of_all_corpus_rows"] = monthly["document_count"] / monthly["month_total_all_corpus_rows"]
    monthly["share_among_assigned_rows"] = np.where(
        monthly["topic_label"].eq("UNASSIGNED"),
        np.nan,
        monthly["document_count"] / monthly["month_total_assigned_rows"].replace(0, np.nan)
    )
    monthly.insert(0, "model", model_key)
    monthly.to_csv(outdir / f"{model_key.lower()}_monthly_topic_prevalence.csv", index=False)

    cm = dated.groupby(["community", "month", "topic_label"], dropna=False).size().rename("document_count").reset_index()
    ctot = dated.groupby(["community", "month"]).size().rename("community_month_total_all_rows").reset_index()
    cas = dated[dated["has_assignment"]].groupby(["community", "month"]).size().rename(
        "community_month_total_assigned"
    ).reset_index()
    cm = cm.merge(ctot, on=["community", "month"], how="left")
    cm = cm.merge(cas, on=["community", "month"], how="left")
    cm["community_month_total_assigned"] = cm["community_month_total_assigned"].fillna(0).astype(int)
    cm["share_of_all_community_month_rows"] = cm["document_count"] / cm["community_month_total_all_rows"]
    cm["share_among_assigned_rows"] = np.where(
        cm["topic_label"].eq("UNASSIGNED"),
        np.nan,
        cm["document_count"] / cm["community_month_total_assigned"].replace(0, np.nan)
    )
    cm.insert(0, "model", model_key)
    cm.to_csv(outdir / f"{model_key.lower()}_community_month_topic_prevalence.csv", index=False)

    overall = dated.groupby(["community", "topic_label"], dropna=False).size().rename("document_count").reset_index()
    totals_c = dated.groupby("community").size().rename("community_total_all_rows").reset_index()
    overall = overall.merge(totals_c, on="community", how="left")
    overall["share_of_all_community_rows"] = overall["document_count"] / overall["community_total_all_rows"]
    overall.insert(0, "model", model_key)
    overall.to_csv(outdir / f"{model_key.lower()}_community_topic_prevalence.csv", index=False)

    merged.to_csv(outdir / f"{model_key.lower()}_document_topic_date_join.csv", index=False)
    assignment_coverage = float(merged["has_assignment"].mean())
    audit = {
        "model": model_key,
        "configuration": str(merged["model_configuration"].dropna().iloc[0]),
        "corpus_rows": int(len(corpus)),
        "assignment_rows_selected": int(len(assignments)),
        "matched_corpus_rows": int(merged["has_assignment"].sum()),
        "unassigned_corpus_rows": int((~merged["has_assignment"]).sum()),
        "assignment_coverage": assignment_coverage,
        "dated_rows": int(merged["analysis_date"].notna().sum()),
        "start_date": str(merged["analysis_date"].min().date()),
        "end_date": str(merged["analysis_date"].max().date()),
        "communities": sorted(merged["community"].dropna().unique().tolist()),
        "months": sorted(merged["month"].dropna().unique().tolist()),
    }
    return audit


def main():
    parser = argparse.ArgumentParser(description="Phase 7O explicit-configuration temporal analysis")
    parser.add_argument("--run-model", action="store_true", help="Run the temporal summaries; default is audit only")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--nmf", type=Path, default=DEFAULT_NMF)
    parser.add_argument("--semantic", type=Path, default=DEFAULT_SEMANTIC)
    parser.add_argument("--nmf-variant", default="unigram_bigram")
    parser.add_argument("--nmf-k", type=int, default=5)
    parser.add_argument("--semantic-k", type=int, default=5)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        corpus = load_corpus(args.corpus)
        print("PHASE 7O CORRECTED AUDIT")
        print(f"Corpus rows: {len(corpus)}")
        print(f"Rows with valid dates: {int(corpus['analysis_date'].notna().sum())}")
        print(f"Date range: {corpus['analysis_date'].min()} to {corpus['analysis_date'].max()}")
        print(f"Communities: {sorted(corpus['community'].unique().tolist())}")

        nmf_raw = require_file(args.nmf, "NMF assignment file")
        sem_raw = require_file(args.semantic, "Semantic assignment file")
        print("\nAvailable NMF configurations:")
        print(nmf_raw.groupby(["variant", "k"], dropna=False).size().to_string())
        print("\nAvailable semantic K configurations:")
        print(sem_raw.groupby(["k"], dropna=False).size().to_string())
        print(f"\nSelected NMF: variant={args.nmf_variant}, K={args.nmf_k}")
        print(f"Selected semantic K: {args.semantic_k}")

        nmf = select_nmf(args.nmf, args.nmf_variant, args.nmf_k)
        semantic = select_semantic(args.semantic, args.semantic_k)
        corpus_ids = set(corpus["record_id"])
        for label, assn in [("NMF", nmf), ("Semantic", semantic)]:
            matched = int(assn["record_id"].isin(corpus_ids).sum())
            print(f"{label}: selected rows={len(assn)}, matched IDs={matched}, unmatched assignment IDs={len(assn)-matched}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(2)

    if not args.run_model:
        print("\nAUDIT ONLY. If configurations and ID matches look correct, rerun with --run-model.")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    audits = []
    for model_key, assignment in [("NMF", nmf), ("Semantic_KMeans", semantic)]:
        audit = make_summaries(corpus, assignment, model_key, args.outdir)
        audits.append(audit)
        print(f"\n{model_key}: matched {audit['matched_corpus_rows']}/{audit['corpus_rows']} "
              f"({audit['assignment_coverage']:.1%}); unassigned={audit['unassigned_corpus_rows']}")
    pd.DataFrame(audits).to_csv(args.outdir / "phase7o_join_audit.csv", index=False)
    manifest = {
        "phase": "7O corrected",
        "corpus": str(args.corpus),
        "nmf_assignment_file": str(args.nmf),
        "semantic_assignment_file": str(args.semantic),
        "selected_nmf": {"variant": args.nmf_variant, "k": args.nmf_k},
        "selected_semantic": {"k": args.semantic_k},
        "audits": audits,
        "interpretation_cautions": [
            "Only one explicit configuration per model is selected before joining.",
            "UNASSIGNED rows remain visible and are included in all-corpus denominators.",
            "Topic labels are model-specific; NMF Topic 1 is not equivalent to semantic Topic 1.",
            "Monthly proportions are descriptive, not causal.",
            "Check month-level sample sizes before interpreting changes."
        ]
    }
    (args.outdir / "phase7o_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nPHASE 7O CORRECTED COMPLETE. Outputs: {args.outdir}")


if __name__ == "__main__":
    main()
