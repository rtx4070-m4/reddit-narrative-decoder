"""Phase 7M: density-based semantic clustering (HDBSCAN).

Run from project root:
    python src/topics/semantic_topic_model_v14.py                  # audit only
    python src/topics/semantic_topic_model_v14.py --run-model       # fit HDBSCAN

Uses the same Phase 7J cleaned text and local sentence-transformer as Phase 7L.
HDBSCAN can leave ambiguous texts unassigned (label -1), rather than forcing every
comment into a cluster. It is an experiment, not guaranteed to outperform KMeans.
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
OUT = ROOT / "results" / "tables" / "phase7m_hdbscan"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_STATE = 42
MIN_TOKENS = 5
BATCH_SIZE = 16
MAX_LENGTH = 256
MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 5

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}. Run Phase 7J first.")
    df = pd.read_csv(INPUT, low_memory=False)
    required = {"record_id", "community", "topic7j_clean_text", "source_autotldr_marker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df["source_autotldr_marker"] = (
        df["source_autotldr_marker"].fillna(False).astype(str).str.lower()
        .isin(["true", "1", "yes"])
    )
    df["semantic_text"] = df["topic7j_clean_text"].fillna("").astype(str).str.strip()
    df["semantic_token_count"] = df["semantic_text"].map(lambda s: len(s.split()))
    df["semantic_eligible"] = df["semantic_token_count"] >= MIN_TOKENS
    df["semantic_fit_eligible"] = df["semantic_eligible"] & ~df["source_autotldr_marker"]
    audit = [
        ("input_rows", len(df)),
        ("eligible_rows_min_5_tokens", int(df["semantic_eligible"].sum())),
        ("excluded_rows_under_5_tokens", int((~df["semantic_eligible"]).sum())),
        ("source_autotldr_rows_excluded_from_fit", int(df["source_autotldr_marker"].sum())),
        ("eligible_non_autotldr_rows_for_fit", int(df["semantic_fit_eligible"].sum())),
        ("eligible_autotldr_rows_for_assignment", int((df["semantic_eligible"] & df["source_autotldr_marker"]).sum())),
        ("unique_eligible_texts", int(df.loc[df["semantic_eligible"], "semantic_text"].nunique())),
        ("unique_non_autotldr_texts_for_fit", int(df.loc[df["semantic_fit_eligible"], "semantic_text"].nunique())),
    ]
    pd.DataFrame(audit, columns=["metric", "value"]).to_csv(OUT / "phase7m_cleaning_audit.csv", index=False)
    sample = df[df["source_autotldr_marker"] | ~df["semantic_eligible"]].copy()
    if len(sample) < 150:
        extra = df.sample(min(150-len(sample), len(df)), random_state=RANDOM_STATE)
        sample = pd.concat([sample, extra], ignore_index=True).drop_duplicates("record_id")
    cols = [c for c in ["record_id", "community", "source_autotldr_marker", "original_text",
                        "semantic_text", "semantic_token_count", "semantic_eligible"] if c in sample.columns]
    sample[cols].head(400).to_csv(OUT / "phase7m_cleaning_examples.csv", index=False)
    print("PHASE 7M AUDIT COMPLETE" + ("" if args.run_model else " (MODEL NOT FITTED)"))
    print(pd.DataFrame(audit, columns=["metric", "value"]).to_string(index=False))
    print(f"Audit files saved to: {OUT}")
    if not args.run_model:
        print("Inspect phase7m_cleaning_examples.csv before fitting.")
        print("When ready: python src/topics/semantic_topic_model_v14.py --run-model")
        return

    try:
        import torch
        from sentence_transformers import SentenceTransformer
        import hdbscan
        from hdbscan import approximate_predict
    except ImportError as exc:
        raise ImportError(
            "Required package missing. Activate .venv and ensure sentence-transformers, "
            "torch, and hdbscan are installed. BERTopic installations commonly include hdbscan."
        ) from exc

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fit_df = df[df["semantic_fit_eligible"]].copy()
    fit_texts = fit_df["semantic_text"].drop_duplicates().tolist()
    eligible_df = df[df["semantic_eligible"]].copy().reset_index(drop=True)
    unique_all_texts = eligible_df["semantic_text"].drop_duplicates().tolist()
    print(f"Loading local model {args.model_name} on {device}; batch size={args.batch_size}")
    encoder = SentenceTransformer(args.model_name, device=device)
    encoder.max_seq_length = MAX_LENGTH
    try:
        emb_unique = encoder.encode(
            unique_all_texts, batch_size=max(1, args.batch_size),
            show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
        )
    except RuntimeError as exc:
        if device == "cuda" and ("out of memory" in str(exc).lower() or "cuda" in str(exc).lower()):
            raise RuntimeError("CUDA memory issue. Retry with --device cpu --batch-size 8.") from exc
        raise
    emb_map = {text: emb_unique[i] for i, text in enumerate(unique_all_texts)}
    E_all = normalize(np.vstack([emb_map[t] for t in eligible_df["semantic_text"]]))
    fit_unique = fit_df["semantic_text"].drop_duplicates().tolist()
    E_fit = normalize(np.vstack([emb_map[t] for t in fit_unique]))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    print(f"Fitting HDBSCAN on {len(fit_unique)} unique non-AutoTLDR texts...")
    fit_labels = clusterer.fit_predict(E_fit)
    all_labels, strengths = approximate_predict(clusterer, E_all)
    # HDBSCAN cluster label -1 means noise/unassigned.
    n_clusters = len(set(fit_labels) - {-1})
    noise_fit_share = float(np.mean(fit_labels == -1))
    noise_all_share = float(np.mean(all_labels == -1))
    cluster_sizes = pd.Series(all_labels).value_counts().sort_index().to_dict()
    summary = pd.DataFrame([
        ("model_name", args.model_name),
        ("device", device),
        ("fit_unique_texts", len(fit_unique)),
        ("eligible_documents_assigned", len(eligible_df)),
        ("min_cluster_size", args.min_cluster_size),
        ("min_samples", args.min_samples),
        ("number_of_fit_clusters_excluding_noise", n_clusters),
        ("fit_noise_share", noise_fit_share),
        ("assigned_noise_share", noise_all_share),
        ("mean_membership_strength", float(np.mean(strengths))),
        ("cluster_counts_assigned", str(cluster_sizes)),
    ], columns=["metric", "value"])
    summary.to_csv(OUT / "phase7m_model_comparison.csv", index=False)

    # Explain clusters with TF-IDF keywords using custom stopwords; clustering itself is semantic.
    vectorizer = TfidfVectorizer(
        stop_words=KEYWORD_STOPS, min_df=2, max_df=0.90,
        ngram_range=(1, 2), max_features=12000, sublinear_tf=True,
    )
    vectorizer.fit(fit_unique)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    X_all = vectorizer.transform(eligible_df["semantic_text"])
    terms_out, community_out, reps_out = [], [], []
    out = eligible_df[[c for c in [
        "record_id", "id", "community", "subreddit", "date", "year",
        "source_file", "original_text", "semantic_text", "semantic_token_count",
        "source_autotldr_marker"
    ] if c in eligible_df.columns]].copy()
    out["cluster"] = all_labels
    out["membership_strength"] = strengths
    out["is_noise"] = all_labels == -1
    out.to_csv(OUT / "phase7m_document_clusters.csv", index=False)

    for cluster in sorted(set(all_labels)):
        idx = np.flatnonzero(all_labels == cluster)
        label_name = "noise_unassigned" if cluster == -1 else f"cluster_{cluster}"
        if len(idx):
            mean_weights = np.asarray(X_all[idx].mean(axis=0)).ravel()
            top_idx = np.argsort(mean_weights)[::-1][:20]
            for rank, term_idx in enumerate(top_idx, start=1):
                terms_out.append({
                    "cluster": int(cluster), "cluster_name": label_name,
                    "rank": rank, "term": vocab[term_idx],
                    "mean_tfidf_weight": float(mean_weights[term_idx]),
                    "document_count": int(len(idx)),
                })
        for community, group in eligible_df.iloc[idx].groupby("community", dropna=False):
            community_out.append({
                "cluster": int(cluster), "cluster_name": label_name,
                "community": str(community), "document_count": int(len(group)),
                "community_documents": int((eligible_df["community"] == community).sum()),
                "within_community_share": float(len(group) / max(1, (eligible_df["community"] == community).sum())),
            })
        # Deduplicate examples by cleaned text so repeated source rows don't fill the list.
        candidates = eligible_df.iloc[idx].copy()
        candidates["cluster"] = all_labels[idx]
        candidates["membership_strength"] = strengths[idx]
        candidates = candidates.drop_duplicates("semantic_text").sort_values(
            "membership_strength", ascending=False
        ).head(12)
        for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
            reps_out.append({
                "cluster": int(cluster), "cluster_name": label_name, "rank": rank,
                "record_id": row.get("record_id", ""), "community": row.get("community", ""),
                "membership_strength": float(row.get("membership_strength", 0)),
                "source_autotldr_marker": row.get("source_autotldr_marker", False),
                "original_text": row.get("original_text", ""),
                "semantic_text": row.get("semantic_text", ""),
            })

    pd.DataFrame(terms_out).to_csv(OUT / "phase7m_topic_terms.csv", index=False)
    pd.DataFrame(community_out).to_csv(OUT / "phase7m_topic_community_distribution.csv", index=False)
    pd.DataFrame(reps_out).to_csv(OUT / "phase7m_representative_documents.csv", index=False)
    (OUT / "phase7m_run_manifest.txt").write_text(
        "Phase 7M HDBSCAN semantic clustering\n"
        f"Input: {INPUT}\nOutput: {OUT}\nModel: {args.model_name}\nDevice: {device}\n"
        f"min_cluster_size={args.min_cluster_size}; min_samples={args.min_samples}\n"
        "Fit uses unique eligible non-AutoTLDR texts; eligible documents are assigned with approximate_predict.\n"
        "Label -1 means HDBSCAN noise/unassigned; it is not a substantive topic.\n",
        encoding="utf-8",
    )
    print("\nPHASE 7M COMPLETE")
    print(summary.to_string(index=False))
    print(f"Outputs saved to: {OUT}")

if __name__ == "__main__":
    main()
