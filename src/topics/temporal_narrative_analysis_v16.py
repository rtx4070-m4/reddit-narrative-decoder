#!/usr/bin/env python
"""
Phase 7O: Temporal narrative analysis for the Reddit narrative decoder project.

This script does NOT refit topic models. It discovers existing document-level topic
assignment CSVs under Phase 7K (NMF) and Phase 7L (semantic K-means), joins them
to the Phase 7J cleaned corpus audit, and produces monthly topic prevalence and
community-by-month summaries.

Because upstream scripts can use different column names, the script:
  * scans candidate CSVs and reports its choices,
  * identifies likely text/date/community/topic columns from common names,
  * uses normalized text as a fallback join key,
  * can be run in --audit-only mode before analysis.

Run from the project root:
  python src/topics/temporal_narrative_analysis_v16.py
  python src/topics/temporal_narrative_analysis_v16.py --run-model

Optional explicit paths:
  --corpus PATH
  --nmf PATH
  --semantic PATH
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
DEFAULT_CORPUS = TABLES / "phase7j_nmf" / "phase7j_cleaned_documents_audit.csv"
DEFAULT_OUT = TABLES / "phase7o_temporal_narratives"

TEXT_NAMES = [
    "analysis_text", "cleaned_text", "text_clean", "document_text",
    "original_text", "text", "document", "content", "body"
]
DATE_NAMES = ["date", "created_date", "created_utc", "timestamp", "created_at", "datetime"]
COMMUNITY_NAMES = ["community", "subreddit", "subreddit_name", "source_community"]
ID_NAMES = ["record_id", "id", "document_id", "doc_id", "post_id"]
TOPIC_NAMES = ["dominant_topic", "topic_id", "topic", "cluster", "cluster_id",
               "predicted_topic", "assigned_topic", "hdbscan_cluster"]
CONFIDENCE_NAMES = ["membership_strength", "topic_probability", "confidence",
                    "max_topic_probability", "dominant_topic_probability"]


def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def col_by_names(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name in lower_map:
            return lower_map[name]
    # conservative partial matching for common variants
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(lc == n or lc.endswith("_" + n) for n in names):
            return c
    return None


def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def discover_assignment_csv(folder: Path, model_kind: str) -> list[tuple[Path, pd.DataFrame, str]]:
    found = []
    if not folder.exists():
        print(f"[WARN] Missing model folder: {folder}")
        return found
    for path in sorted(folder.rglob("*.csv")):
        # Ignore terms, representatives, aggregate distributions and manifests.
        stem = path.stem.lower()
        if any(token in stem for token in (
            "term", "representative", "distribution", "comparison", "manifest",
            "audit", "example", "summary", "metric", "stability", "coherence"
        )):
            continue
        df = read_csv_safe(path)
        if df is None or len(df) < 10:
            continue
        topic_col = col_by_names(df, TOPIC_NAMES)
        text_col = col_by_names(df, TEXT_NAMES)
        id_col = col_by_names(df, ID_NAMES)
        if topic_col is None:
            # Topic-probability matrices sometimes contain Topic_1, Topic_2...
            topic_like = [c for c in df.columns if re.match(r"(?i)^(topic|cluster)[ _-]?\d+$", str(c).strip())]
            if len(topic_like) >= 2:
                found.append((path, df, "wide_topic_matrix"))
            continue
        if text_col is None and id_col is None:
            continue
        if model_kind == "nmf" and not any(k in stem for k in ("document", "assignment", "topic", "cluster")):
            # Still retain if it clearly has row-level topic and text/id columns.
            pass
        found.append((path, df, "row_assignment"))
    return found


def score_candidate(item, corpus_texts: set[str], corpus_ids: set[str]) -> float:
    path, df, kind = item
    score = 0.0
    stem = path.stem.lower()
    if "document" in stem: score += 5
    if "topic" in stem or "assignment" in stem or "cluster" in stem: score += 2
    text_col = col_by_names(df, TEXT_NAMES)
    id_col = col_by_names(df, ID_NAMES)
    if text_col:
        sample = df[text_col].head(300).map(norm_text)
        overlap = sum(x in corpus_texts for x in sample if x)
        score += min(10, overlap / 10)
    if id_col:
        sample = df[id_col].astype(str).head(300)
        overlap = sum(x in corpus_ids for x in sample if x)
        score += min(10, overlap / 10)
    if kind == "wide_topic_matrix": score -= 2
    return score


def prepare_corpus(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {path}\n"
            "Expected Phase 7J cleaned audit CSV. Use --corpus to provide its path."
        )
    df = pd.read_csv(path, low_memory=False)
    text_col = col_by_names(df, TEXT_NAMES)
    if text_col is None:
        raise ValueError(f"No recognizable text column in corpus file. Columns: {list(df.columns)}")
    df = df.copy()
    if text_col != "analysis_text":
        df["analysis_text"] = df[text_col]
    date_col = col_by_names(df, DATE_NAMES)
    if date_col is None:
        # Build a date from year/month/day if available.
        lower = {str(c).lower(): c for c in df.columns}
        if "year" in lower and "month" in lower:
            day_col = lower.get("day")
            day = df[day_col] if day_col else 1
            df["analysis_date"] = pd.to_datetime(
                {"year": pd.to_numeric(df[lower["year"]], errors="coerce"),
                 "month": pd.to_numeric(df[lower["month"]], errors="coerce"),
                 "day": pd.to_numeric(day, errors="coerce")},
                errors="coerce"
            )
        else:
            raise ValueError(f"No date or year/month columns found. Columns: {list(df.columns)}")
    else:
        # Epoch-like numeric timestamps are parsed as seconds when plausible.
        raw = df[date_col]
        if pd.api.types.is_numeric_dtype(raw):
            numeric = pd.to_numeric(raw, errors="coerce")
            median = numeric.dropna().median() if numeric.notna().any() else np.nan
            unit = "s" if pd.notna(median) and median > 100_000_000 else "ms"
            df["analysis_date"] = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True).dt.tz_convert(None)
        else:
            df["analysis_date"] = pd.to_datetime(raw, errors="coerce", utc=True).dt.tz_convert(None)
    comm_col = col_by_names(df, COMMUNITY_NAMES)
    if comm_col is None:
        raise ValueError(f"No community/subreddit column found. Columns: {list(df.columns)}")
    if comm_col != "analysis_community":
        df["analysis_community"] = df[comm_col].astype(str)
    id_col = col_by_names(df, ID_NAMES)
    df["analysis_id"] = df[id_col].astype(str) if id_col else ""
    df["analysis_text_key"] = df["analysis_text"].map(norm_text)
    df["analysis_month"] = df["analysis_date"].dt.to_period("M").astype(str)
    # Restrict to eligible rows if there is a clear boolean/flag column; otherwise preserve rows.
    eligible_col = col_by_names(df, ["eligible", "is_eligible", "eligible_min_5_tokens"])
    if eligible_col:
        vals = df[eligible_col].astype(str).str.lower()
        true_vals = {"true", "1", "yes", "eligible"}
        if vals.isin(true_vals).any():
            df = df[vals.isin(true_vals)].copy()
    return df


def assignment_to_long(path: Path, df: pd.DataFrame, kind: str) -> Optional[pd.DataFrame]:
    topic_col = col_by_names(df, TOPIC_NAMES)
    text_col = col_by_names(df, TEXT_NAMES)
    id_col = col_by_names(df, ID_NAMES)
    conf_col = col_by_names(df, CONFIDENCE_NAMES)
    if kind == "wide_topic_matrix":
        topic_cols = [c for c in df.columns if re.match(r"(?i)^(topic|cluster)[ _-]?\d+$", str(c).strip())]
        values = df[topic_cols].apply(pd.to_numeric, errors="coerce")
        dominant = values.idxmax(axis=1)
        out = pd.DataFrame({"topic_label": dominant.astype(str)})
        out["topic_score"] = values.max(axis=1)
        if text_col: out["analysis_text_key"] = df[text_col].map(norm_text)
        if id_col: out["analysis_id"] = df[id_col].astype(str)
        return out
    if topic_col is None:
        return None
    out = pd.DataFrame({"topic_label": df[topic_col].astype(str)})
    if text_col:
        out["analysis_text_key"] = df[text_col].map(norm_text)
    if id_col:
        out["analysis_id"] = df[id_col].astype(str)
    if conf_col:
        out["topic_score"] = pd.to_numeric(df[conf_col], errors="coerce")
    return out


def join_assignments(corpus: pd.DataFrame, assignment: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    left = corpus.copy()
    right = assignment.copy()
    if "analysis_id" in right and "analysis_id" in left and (right["analysis_id"] != "").any():
        # Use ID only when overlap is substantial and IDs are not just empty/default.
        left_ids = set(left["analysis_id"].astype(str))
        right_ids = set(right["analysis_id"].astype(str))
        overlap = len(left_ids & right_ids) / max(1, len(right_ids))
        if overlap >= 0.25:
            right = right.drop_duplicates("analysis_id", keep="first")
            merged = left.merge(right, on="analysis_id", how="left", suffixes=("", "_model"))
            if merged["topic_label"].notna().mean() >= 0.20:
                return merged, "analysis_id"
    if "analysis_text_key" in right:
        right = right[right["analysis_text_key"].astype(str).ne("")].drop_duplicates("analysis_text_key", keep="first")
        merged = left.merge(right, on="analysis_text_key", how="left", suffixes=("", "_model"))
        return merged, "normalized text"
    raise ValueError("Could not join topic assignments: no usable ID or text key.")


def write_outputs(corpus: pd.DataFrame, model_name: str, joined: pd.DataFrame, join_key: str, outdir: Path):
    joined = joined.copy()
    joined["model"] = model_name
    joined["topic_label"] = joined["topic_label"].fillna("UNASSIGNED").astype(str)
    # HDBSCAN noise should be retained as a special non-topic, but do not confuse it with a real topic.
    joined["topic_label"] = joined["topic_label"].replace({"-1": "NOISE", "nan": "UNASSIGNED", "None": "UNASSIGNED"})
    joined["month"] = joined["analysis_date"].dt.to_period("M").astype(str)
    valid_dates = joined["analysis_date"].notna()
    dated = joined[valid_dates].copy()
    if dated.empty:
        raise ValueError(f"No valid dates after joining {model_name} assignments. Check join columns.")
    dated["is_assigned_topic"] = ~dated["topic_label"].isin(["NOISE", "UNASSIGNED", "-1"])
    # Monthly counts and within-month prevalence, including unassigned/noise as explicit categories.
    monthly = dated.groupby(["model", "month", "topic_label"], dropna=False).size().rename("document_count").reset_index()
    totals = dated.groupby(["model", "month"]).size().rename("month_total").reset_index()
    monthly = monthly.merge(totals, on=["model", "month"], how="left")
    monthly["within_month_share"] = monthly["document_count"] / monthly["month_total"]
    monthly.to_csv(outdir / f"{model_name}_monthly_topic_prevalence.csv", index=False)

    community = dated.groupby(["model", "analysis_community", "month", "topic_label"], dropna=False).size().rename("document_count").reset_index()
    comm_totals = dated.groupby(["model", "analysis_community", "month"]).size().rename("community_month_total").reset_index()
    community = community.merge(comm_totals, on=["model", "analysis_community", "month"], how="left")
    community["within_community_month_share"] = community["document_count"] / community["community_month_total"]
    community.to_csv(outdir / f"{model_name}_community_month_topic_prevalence.csv", index=False)

    community_overall = dated.groupby(["model", "analysis_community", "topic_label"], dropna=False).size().rename("document_count").reset_index()
    overall_totals = dated.groupby(["model", "analysis_community"]).size().rename("community_total").reset_index()
    community_overall = community_overall.merge(overall_totals, on=["model", "analysis_community"], how="left")
    community_overall["within_community_share"] = community_overall["document_count"] / community_overall["community_total"]
    community_overall.to_csv(outdir / f"{model_name}_community_topic_prevalence.csv", index=False)

    joined.to_csv(outdir / f"{model_name}_document_topic_date_join.csv", index=False)
    join_audit = {
        "model": model_name, "join_key": join_key, "corpus_rows": int(len(corpus)),
        "joined_rows": int(len(joined)), "dated_rows": int(valid_dates.sum()),
        "topic_assignment_coverage": float(joined["topic_label"].ne("UNASSIGNED").mean()),
        "nonnoise_topic_coverage": float(joined["topic_label"].isin(["NOISE", "UNASSIGNED"]).eq(False).mean()),
        "months": sorted(dated["month"].dropna().unique().tolist()),
        "communities": sorted(dated["analysis_community"].dropna().unique().tolist())
    }
    return join_audit


def main():
    parser = argparse.ArgumentParser(description="Phase 7O temporal narrative analysis")
    parser.add_argument("--run-model", action="store_true", help="Run topic assignment joins and temporal summaries")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Phase 7J cleaned corpus audit CSV")
    parser.add_argument("--nmf", type=Path, default=TABLES / "phase7k_nmf", help="Phase 7K NMF output folder or assignment CSV")
    parser.add_argument("--semantic", type=Path, default=TABLES / "phase7l_semantic", help="Phase 7L semantic output folder or assignment CSV")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT, help="Output folder")
    args = parser.parse_args()

    print("PHASE 7O TEMPORAL NARRATIVE AUDIT")
    try:
        corpus = prepare_corpus(args.corpus)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(2)
    corpus_texts = set(corpus["analysis_text_key"].dropna().astype(str))
    corpus_ids = set(corpus["analysis_id"].dropna().astype(str))
    print(f"Corpus: {args.corpus}")
    print(f"Rows loaded: {len(corpus)}")
    print(f"Rows with valid dates: {int(corpus['analysis_date'].notna().sum())}")
    print(f"Communities: {corpus['analysis_community'].nunique()}")
    print(f"Date range: {corpus['analysis_date'].min()} to {corpus['analysis_date'].max()}")
    print(f"Normalized nonempty text keys: {sum(corpus['analysis_text_key'].ne(''))}")

    candidates_by_model = {}
    for name, path in [("nmf", args.nmf), ("semantic", args.semantic)]:
        if path.is_file():
            folder = path.parent
            explicit = read_csv_safe(path)
            candidates = []
            if explicit is not None:
                topic_col = col_by_names(explicit, TOPIC_NAMES)
                text_col = col_by_names(explicit, TEXT_NAMES)
                id_col = col_by_names(explicit, ID_NAMES)
                wide = any(re.match(r"(?i)^(topic|cluster)[ _-]?\d+$", str(c).strip()) for c in explicit.columns)
                if (topic_col and (text_col or id_col)) or (wide and (text_col or id_col)):
                    candidates.append((path, explicit, "wide_topic_matrix" if (topic_col is None and wide) else "row_assignment"))
        else:
            candidates = discover_assignment_csv(path, name)
        scored = sorted([(score_candidate(x, corpus_texts, corpus_ids), x) for x in candidates],
                        key=lambda pair: pair[0], reverse=True)
        candidates_by_model[name] = scored
        print(f"\n{name.upper()} assignment candidates:")
        for score, (p, df, kind) in scored[:5]:
            print(f"  score={score:.2f} rows={len(df)} type={kind}: {p}")
        if scored:
            print(f"  Selected: {scored[0][1][0]}")
        else:
            print("  No row-level assignment CSV found.")

    if not args.run_model:
        print("\nAUDIT ONLY. Inspect the selected candidate paths above.")
        print("If a wrong CSV was selected, rerun with --nmf PATH and/or --semantic PATH.")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    audits = []
    for name, scored in candidates_by_model.items():
        if not scored:
            print(f"[WARN] Skipping {name}: no suitable assignment CSV discovered.")
            continue
        _, (path, df, kind) = scored[0]
        assignment = assignment_to_long(path, df, kind)
        if assignment is None:
            print(f"[WARN] Skipping {name}: could not parse topic assignments from {path}")
            continue
        joined, join_key = join_assignments(corpus, assignment)
        audit = write_outputs(corpus, name, joined, join_key, args.outdir)
        audit["assignment_file"] = str(path)
        audit["assignment_rows"] = int(len(df))
        audits.append(audit)
        print(f"\n{name.upper()} temporal summaries written. Join key: {join_key}")
        print(f"  Joined rows: {audit['joined_rows']}; dated rows: {audit['dated_rows']}")
        print(f"  Assignment coverage: {audit['topic_assignment_coverage']:.3f}")
        print(f"  Non-noise topic coverage: {audit['nonnoise_topic_coverage']:.3f}")
    if audits:
        pd.DataFrame(audits).to_csv(args.outdir / "phase7o_join_audit.csv", index=False)
        manifest = {
            "phase": "7O", "description": "Exploratory temporal topic prevalence by month and community",
            "corpus": str(args.corpus), "output_dir": str(args.outdir),
            "models": audits,
            "cautions": [
                "Topic labels are model-specific and are not directly comparable between NMF and semantic K-means.",
                "Monthly shares are descriptive, not causal.",
                "Low monthly counts can make prevalence estimates unstable.",
                "If assignments are joined by normalized text, identical texts can share an assignment; duplicates remain separate corpus observations.",
                "The source corpus is concentrated in 2016; avoid claims of long-term temporal evolution without adequate observations."
            ]
        }
        (args.outdir / "phase7o_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nPHASE 7O COMPLETE. Outputs: {args.outdir}")
    else:
        print("\nNo temporal summaries were created. Review candidate paths and assignment column names.")


if __name__ == "__main__":
    main()
