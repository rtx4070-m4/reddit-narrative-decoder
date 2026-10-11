"""Phase 7N: HDBSCAN parameter sensitivity for semantic topic clustering.

Run from the Reddit project root:
    python src/topics/semantic_topic_sensitivity_v15.py                 # audit only
    python src/topics/semantic_topic_sensitivity_v15.py --run-model      # fit parameter grid
    python src/topics/semantic_topic_sensitivity_v15.py --run-model --device cpu --batch-size 8

Embeds eligible texts once, then compares fixed HDBSCAN configurations. Noise label -1
means unassigned, not a topic. Results are diagnostic and should be judged by both
coverage and representative-text coherence, not by noise share alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables" / "phase7j_nmf" / "phase7j_cleaned_documents_audit.csv"
OUT = ROOT / "results" / "tables" / "phase7n_hdbscan_sensitivity"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_TOKENS = 5
MAX_LENGTH = 256
BATCH_SIZE = 16
CONFIGS = [
    ("A_mcs15_ms1", 15, 1),
    ("B_mcs15_ms3", 15, 3),
    ("C_mcs30_ms1", 30, 1),
    ("D_mcs30_ms3", 30, 3),
    ("E_mcs30_ms5_baseline", 30, 5),
]
CUSTOM_STOP = {
    "like", "just", "know", "want", "really", "thing", "things", "actually",
    "basically", "literally", "anyway", "anyways", "yeah", "yep", "nah",
    "lol", "lmao", "omg", "thanks", "thank", "please", "sorry", "im", "ive",
    "id", "ill", "youre", "theyre", "weve", "thats", "theres", "heres",
    "whats", "lets", "aint", "http", "https", "www", "com", "org", "net",
    "amp", "nbsp", "gt", "ve", "re", "nt", "autotldr", "tldr", "original",
    "reduced", "keywords", "admin", "moderators", "version", "summary",
}
KEYWORD_STOPS = sorted((set(ENGLISH_STOP_WORDS) - {"no", "nor", "not"}) | CUSTOM_STOP)


def bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-model", action="store_true", help="Fit all HDBSCAN configurations")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--min-cluster-size", type=int, default=None,
                        help="Optional single min_cluster_size override; default uses the 5-setting grid")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="Optional single min_samples override; requires --min-cluster-size")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}. Run Phase 7J first.")
    df = pd.read_csv(INPUT, low_memory=False)
    required = {"record_id", "community", "topic7j_clean_text", "source_autotldr_marker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df["source_autotldr_marker"] = bool_series(df["source_autotldr_marker"])
    df["semantic_text"] = df["topic7j_clean_text"].fillna("").astype(str).str.strip()
    df["semantic_token_count"] = df["semantic_text"].map(lambda s: len(s.split()))
    df["semantic_eligible"] = df["semantic_token_count"] >= MIN_TOKENS
    df["semantic_fit_eligible"] = df["semantic_eligible"] & ~df["source_autotldr_marker"]
    eligible = df[df["semantic_eligible"]].copy().reset_index(drop=True)
    fit_df = df[df["semantic_fit_eligible"]].copy()
    fit_texts = fit_df["semantic_text"].drop_duplicates().tolist()
    all_texts = eligible["semantic_text"].drop_duplicates().tolist()
    audit = pd.DataFrame([
        ("input_rows", len(df)),
        ("eligible_rows_min_5_tokens", int(df["semantic_eligible"].sum())),
        ("excluded_rows_under_5_tokens", int((~df["semantic_eligible"]).sum())),
        ("source_autotldr_rows_excluded_from_fit", int(df["source_autotldr_marker"].sum())),
        ("eligible_non_autotldr_rows_for_fit", int(df["semantic_fit_eligible"].sum())),
        ("eligible_autotldr_rows_for_assignment", int((df["semantic_eligible"] & df["source_autotldr_marker"]).sum())),
        ("unique_eligible_texts", int(eligible["semantic_text"].nunique())),
        ("unique_non_autotldr_texts_for_fit", len(fit_texts)),
    ], columns=["metric", "value"])
    audit.to_csv(OUT / "phase7n_cleaning_audit.csv", index=False)
    print("PHASE 7N AUDIT COMPLETE" + ("" if args.run_model else " (MODEL NOT FITTED)"))
    print(audit.to_string(index=False))
    print(f"Outputs: {OUT}")
    if not args.run_model:
        print("Review the audit, then run with --run-model to fit the sensitivity grid.")
        return

    try:
        import torch
        import hdbscan
        from hdbscan import approximate_predict
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Missing dependency. Activate .venv and ensure torch, sentence-transformers, and hdbscan are installed.") from exc

    if (args.min_cluster_size is None) != (args.min_samples is None):
        parser.error("Use both --min-cluster-size and --min-samples together for a single custom setting.")
    configs = ([ (f"custom_mcs{args.min_cluster_size}_ms{args.min_samples}", args.min_cluster_size, args.min_samples) ]
               if args.min_cluster_size is not None else CONFIGS)
    if any(mcs < 2 or ms < 1 for _, mcs, ms in configs):
        parser.error("min-cluster-size must be >= 2 and min-samples must be >= 1")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model_name} on {device}; encoding {len(all_texts)} unique eligible texts once...")
    encoder = SentenceTransformer(args.model_name, device=device)
    encoder.max_seq_length = MAX_LENGTH
    try:
        embeddings = encoder.encode(all_texts, batch_size=max(1, args.batch_size),
                                    show_progress_bar=True, convert_to_numpy=True,
                                    normalize_embeddings=True)
    except RuntimeError as exc:
        if device == "cuda" and ("out of memory" in str(exc).lower() or "cuda" in str(exc).lower()):
            raise RuntimeError("CUDA memory issue. Retry with --device cpu --batch-size 8.") from exc
        raise
    emb_map = {txt: embeddings[i] for i, txt in enumerate(all_texts)}
    E_all = normalize(np.vstack([emb_map[t] for t in eligible["semantic_text"]]))
    E_fit = normalize(np.vstack([emb_map[t] for t in fit_texts]))

    vectorizer = TfidfVectorizer(stop_words=KEYWORD_STOPS, min_df=2, max_df=0.90,
                                 ngram_range=(1, 2), max_features=12000, sublinear_tf=True)
    vectorizer.fit(fit_texts)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    X_all = vectorizer.transform(eligible["semantic_text"])

    summaries, cluster_rows, term_rows, rep_rows, community_rows = [], [], [], [], []
    for config_name, mcs, ms in configs:
        print(f"\nFitting {config_name}: min_cluster_size={mcs}, min_samples={ms}")
        model = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                                cluster_selection_method="eom", prediction_data=True, core_dist_n_jobs=1)
        fit_labels = model.fit_predict(E_fit)
        labels, strengths = approximate_predict(model, E_all)
        fit_cluster_ids = sorted(set(fit_labels) - {-1})
        counts = pd.Series(labels).value_counts().sort_index()
        n_noise_fit = float(np.mean(fit_labels == -1))
        n_noise_all = float(np.mean(labels == -1))
        assigned_mask = labels != -1
        summary = {
            "configuration": config_name, "min_cluster_size": mcs, "min_samples": ms,
            "fit_unique_texts": len(fit_texts), "eligible_documents": len(eligible),
            "fit_clusters_excluding_noise": len(fit_cluster_ids),
            "fit_noise_share": n_noise_fit, "assigned_noise_share": n_noise_all,
            "assigned_nonnoise_share": 1.0 - n_noise_all,
            "assigned_nonnoise_documents": int(assigned_mask.sum()),
            "mean_membership_strength_all": float(np.mean(strengths)),
            "mean_membership_strength_nonnoise": float(np.mean(strengths[assigned_mask])) if assigned_mask.any() else np.nan,
            "smallest_nonnoise_cluster": int(min([int(counts.get(c, 0)) for c in fit_cluster_ids], default=0)),
            "largest_assigned_cluster": int(max([int(counts.get(c, 0)) for c in fit_cluster_ids], default=0)),
            "cluster_counts_assigned": str({int(k): int(v) for k, v in counts.items()}),
        }
        summaries.append(summary)
        labels_df = eligible[[c for c in ["record_id", "id", "community", "subreddit", "date", "year",
                                              "source_file", "original_text", "semantic_text", "semantic_token_count",
                                              "source_autotldr_marker"] if c in eligible.columns]].copy()
        labels_df["configuration"] = config_name
        labels_df["min_cluster_size"] = mcs
        labels_df["min_samples"] = ms
        labels_df["cluster"] = labels
        labels_df["membership_strength"] = strengths
        labels_df["is_noise"] = labels == -1
        labels_df.to_csv(OUT / f"phase7n_clusters_{config_name}.csv", index=False)
        for cluster_id, count in counts.items():
            cluster_rows.append({"configuration": config_name, "cluster": int(cluster_id),
                                 "cluster_name": "noise_unassigned" if int(cluster_id) == -1 else f"cluster_{int(cluster_id)}",
                                 "document_count": int(count),
                                 "document_share": float(count / max(1, len(eligible)))})

        for cluster in sorted(set(labels)):
            idx = np.flatnonzero(labels == cluster)
            cname = "noise_unassigned" if cluster == -1 else f"cluster_{cluster}"
            counts_by_comm = eligible.iloc[idx].groupby("community", dropna=False).size()
            for comm, count in counts_by_comm.items():
                community_rows.append({"configuration": config_name, "cluster": int(cluster),
                                       "cluster_name": cname, "community": str(comm),
                                       "document_count": int(count),
                                       "within_community_share": float(count / max(1, (eligible["community"] == comm).sum()))})
            if cluster != -1 and len(idx):
                mean_weights = np.asarray(X_all[idx].mean(axis=0)).ravel()
                for rank, term_idx in enumerate(np.argsort(mean_weights)[::-1][:20], start=1):
                    term_rows.append({"configuration": config_name, "cluster": int(cluster),
                                      "cluster_name": cname, "rank": rank, "term": vocab[term_idx],
                                      "mean_tfidf_weight": float(mean_weights[term_idx]), "document_count": len(idx)})
            candidates = eligible.iloc[idx].copy()
            candidates["cluster"] = labels[idx]
            candidates["membership_strength"] = strengths[idx]
            candidates = candidates.drop_duplicates("semantic_text").sort_values("membership_strength", ascending=False).head(10)
            for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
                rep_rows.append({"configuration": config_name, "cluster": int(cluster), "cluster_name": cname,
                                 "rank": rank, "record_id": row.get("record_id", ""),
                                 "community": row.get("community", ""),
                                 "membership_strength": float(row.get("membership_strength", 0)),
                                 "source_autotldr_marker": row.get("source_autotldr_marker", False),
                                 "original_text": row.get("original_text", ""),
                                 "semantic_text": row.get("semantic_text", "")})

    pd.DataFrame(summaries).to_csv(OUT / "phase7n_sensitivity_comparison.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(OUT / "phase7n_cluster_counts.csv", index=False)
    pd.DataFrame(term_rows).to_csv(OUT / "phase7n_topic_terms.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OUT / "phase7n_representative_documents.csv", index=False)
    pd.DataFrame(community_rows).to_csv(OUT / "phase7n_topic_community_distribution.csv", index=False)
    (OUT / "phase7n_run_manifest.txt").write_text(
        "Phase 7N HDBSCAN sensitivity analysis\n"
        f"Input: {INPUT}\nOutput: {OUT}\nModel: {args.model_name}\nDevice: {device}\n"
        f"Configurations: {configs}\n"
        "Embeddings computed once and reused across configurations. Fit on unique eligible non-AutoTLDR texts; all eligible rows assigned with approximate_predict.\n"
        "Cluster label -1 is noise/unassigned, not a substantive topic. Choose based on coverage AND manual coherence review.\n",
        encoding="utf-8")
    print("\nPHASE 7N COMPLETE")
    print(pd.DataFrame(summaries)[["configuration", "min_cluster_size", "min_samples",
          "fit_clusters_excluding_noise", "assigned_noise_share", "assigned_nonnoise_share",
          "mean_membership_strength_nonnoise", "cluster_counts_assigned"]].to_string(index=False))
    print(f"Outputs saved to: {OUT}")

if __name__ == "__main__":
    main()
