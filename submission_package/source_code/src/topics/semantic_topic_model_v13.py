"""Phase 7L: local semantic topic-model sensitivity experiment.

Run from project root:
    python src/topics/semantic_topic_model_v13.py                 # audit only
    python src/topics/semantic_topic_model_v13.py --run-model      # embeddings + clustering

This phase compares KMeans clustering of local sentence embeddings at K=5 and K=6.
It uses Phase 7J's cleaned corpus and preserves all prior NMF outputs. No Reddit or
LLM API is used. The sentence-transformers model may need to be downloaded once if
it is not already cached locally.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables" / "phase7j_nmf" / "phase7j_cleaned_documents_audit.csv"
OUT = ROOT / "results" / "tables" / "phase7l_semantic"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_VALUES = [5, 6]
MIN_TOKENS = 5
RANDOM_STATE = 42
BATCH_SIZE = 16  # Conservative for a 4 GB laptop GPU; can lower to 8 if needed.
MAX_LENGTH = 256

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run Phase 7J first; no prior outputs are overwritten."
        )

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
        ("eligible_autotldr_rows_transform_only",
         int((df["semantic_eligible"] & df["source_autotldr_marker"]).sum())),
        ("unique_eligible_texts", int(df.loc[df["semantic_eligible"], "semantic_text"].nunique())),
        ("unique_non_autotldr_texts_for_fit",
         int(df.loc[df["semantic_fit_eligible"], "semantic_text"].nunique())),
    ]
    pd.DataFrame(audit, columns=["metric", "value"]).to_csv(
        OUT / "phase7l_cleaning_audit.csv", index=False
    )
    sample = df[df["source_autotldr_marker"] | ~df["semantic_eligible"]].copy()
    if len(sample) < 150:
        extra = df.sample(min(150 - len(sample), len(df)), random_state=RANDOM_STATE)
        sample = pd.concat([sample, extra], ignore_index=True).drop_duplicates("record_id")
    sample_cols = [c for c in [
        "record_id", "community", "source_autotldr_marker", "original_text",
        "topic7j_clean_text", "semantic_text", "semantic_token_count",
        "semantic_eligible", "semantic_fit_eligible"
    ] if c in sample.columns]
    sample[sample_cols].head(400).to_csv(OUT / "phase7l_cleaning_examples.csv", index=False)
    df.to_csv(OUT / "phase7l_documents_audit.csv", index=False)

    print("PHASE 7L SEMANTIC AUDIT COMPLETE" + ("" if args.run_model else " (MODEL NOT FITTED)"))
    print(pd.DataFrame(audit, columns=["metric", "value"]).to_string(index=False))
    print(f"Audit files saved to: {OUT}")
    if not args.run_model:
        print("Inspect phase7l_cleaning_examples.csv before running embeddings.")
        print("When ready: python src/topics/semantic_topic_model_v13.py --run-model")
        return

    # Fit on unique eligible non-AutoTLDR texts, then assign all eligible rows.
    fit_df = df[df["semantic_fit_eligible"]].copy()
    fit_texts = fit_df["semantic_text"].drop_duplicates().tolist()
    all_df = df[df["semantic_eligible"]].copy().reset_index(drop=True)
    if len(fit_texts) < 20:
        raise ValueError("Too few unique fit texts for semantic clustering.")

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Missing dependencies. Activate your project .venv and install "
            "'sentence-transformers' and 'torch' in that environment."
        ) from exc

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading local embedding model: {args.model_name}")
    print(f"Selected device: {device}; batch size: {args.batch_size}")
    encoder = SentenceTransformer(args.model_name, device=device)
    encoder.max_seq_length = MAX_LENGTH

    # Encode unique strings once and reuse embeddings for repeated documents.
    unique_all_texts = all_df["semantic_text"].drop_duplicates().tolist()
    print(f"Encoding {len(unique_all_texts)} unique eligible texts...")
    try:
        unique_embeddings = encoder.encode(
            unique_all_texts, batch_size=max(1, args.batch_size),
            show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except RuntimeError as exc:
        if device == "cuda" and ("out of memory" in str(exc).lower() or "cuda" in str(exc).lower()):
            raise RuntimeError(
                "Embedding failed on CUDA. Retry with "
                "--device cpu --batch-size 8. No results were overwritten beyond audit files."
            ) from exc
        raise

    embedding_map = {text: unique_embeddings[i] for i, text in enumerate(unique_all_texts)}
    E_all = np.vstack([embedding_map[t] for t in all_df["semantic_text"]])
    fit_unique = fit_df["semantic_text"].drop_duplicates().tolist()
    fit_embeddings = np.vstack([embedding_map[t] for t in fit_unique])
    # KMeans uses Euclidean distance; normalized vectors make distance broadly aligned
    # with cosine similarity for comparison and representative-document selection.
    fit_embeddings = normalize(fit_embeddings)
    E_all = normalize(E_all)

    # Lexical representation is used only to explain clusters, not to create them.
    repr_vectorizer = TfidfVectorizer(
        stop_words="english", min_df=2, max_df=0.90,
        ngram_range=(1, 2), max_features=12000, sublinear_tf=True,
    )
    repr_vectorizer.fit(fit_unique)
    vocab = np.asarray(repr_vectorizer.get_feature_names_out())
    X_repr_all = repr_vectorizer.transform(all_df["semantic_text"])
    X_repr_fit = repr_vectorizer.transform(fit_unique)

    comparisons, term_rows, doc_rows, community_rows, rep_rows = [], [], [], [], []
    for k in K_VALUES:
        print(f"Fitting semantic KMeans K={k}...")
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
        fit_labels = model.fit_predict(fit_embeddings)
        all_labels = model.predict(E_all)
        centroids = normalize(model.cluster_centers_)
        cosine_like_distance = 1.0 - np.sum(E_all * centroids[all_labels], axis=1)
        # Silhouette on a deterministic sample avoids excessive computation.
        sample_n = min(1500, len(fit_embeddings))
        if sample_n > k:
            rng = np.random.default_rng(RANDOM_STATE)
            sample_idx = rng.choice(len(fit_embeddings), size=sample_n, replace=False)
            try:
                sil = float(silhouette_score(
                    fit_embeddings[sample_idx], fit_labels[sample_idx],
                    metric="cosine", sample_size=None,
                ))
            except Exception:
                sil = float("nan")
        else:
            sil = float("nan")

        counts = pd.Series(all_labels + 1).value_counts().sort_index().to_dict()
        comparisons.append({
            "method": "sentence_embedding_kmeans",
            "model_name": args.model_name, "k": k, "fit_unique_texts": len(fit_unique),
            "eligible_documents_assigned": len(all_df), "silhouette_cosine_sample": sil,
            "mean_centroid_cosine_distance": float(np.mean(cosine_like_distance)),
            "dominant_cluster_counts": str(counts),
            "fit_cluster_counts": str(pd.Series(fit_labels + 1).value_counts().sort_index().to_dict()),
        })

        # Cluster terms from average TF-IDF weights among assigned eligible documents.
        for cluster in range(k):
            row_idx = np.flatnonzero(all_labels == cluster)
            if len(row_idx):
                mean_weights = np.asarray(X_repr_all[row_idx].mean(axis=0)).ravel()
                top_idx = np.argsort(mean_weights)[::-1][:20]
                for rank, term_idx in enumerate(top_idx, start=1):
                    term_rows.append({
                        "k": k, "topic": cluster + 1, "rank": rank,
                        "term": vocab[term_idx], "mean_tfidf_weight": float(mean_weights[term_idx]),
                        "cluster_document_count": int(len(row_idx)),
                    })

        keep = [c for c in [
            "record_id", "id", "community", "subreddit", "date", "year",
            "source_file", "original_text", "semantic_text", "semantic_token_count",
            "source_autotldr_marker"
        ] if c in all_df.columns]
        out = all_df[keep].copy()
        out.insert(0, "k", k)
        out["topic"] = all_labels + 1
        out["centroid_cosine_distance"] = cosine_like_distance
        doc_rows.append(out)

        for community, group in out.groupby("community", dropna=False):
            for cluster in range(1, k + 1):
                community_rows.append({
                    "k": k, "community": str(community), "topic": cluster,
                    "document_count": int((group["topic"] == cluster).sum()),
                    "community_documents": len(group),
                    "within_community_share": float((group["topic"] == cluster).mean()),
                })

        for cluster in range(k):
            candidate_indices = np.flatnonzero(all_labels == cluster)
            candidate_indices = candidate_indices[
                np.argsort(cosine_like_distance[candidate_indices])
            ][:10]
            for rank, idx in enumerate(candidate_indices, start=1):
                row = all_df.iloc[idx]
                rep_rows.append({
                    "k": k, "topic": cluster + 1, "rank": rank,
                    "record_id": row.get("record_id", ""),
                    "community": row.get("community", ""),
                    "source_autotldr_marker": row.get("source_autotldr_marker", False),
                    "centroid_cosine_distance": float(cosine_like_distance[idx]),
                    "original_text": row.get("original_text", ""),
                    "semantic_text": row.get("semantic_text", ""),
                })

    pd.DataFrame(comparisons).to_csv(OUT / "phase7l_model_comparison.csv", index=False)
    pd.DataFrame(term_rows).to_csv(OUT / "phase7l_topic_terms.csv", index=False)
    pd.concat(doc_rows, ignore_index=True).to_csv(OUT / "phase7l_document_topics.csv", index=False)
    pd.DataFrame(community_rows).to_csv(OUT / "phase7l_topic_community_distribution.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OUT / "phase7l_representative_documents.csv", index=False)
    (OUT / "phase7l_run_manifest.txt").write_text(
        "Phase 7L local semantic topic experiment\n"
        f"Input: {INPUT}\nOutput: {OUT}\nEmbedding model: {args.model_name}\n"
        f"Device: {device}\nK values: {K_VALUES}\nRandom state: {RANDOM_STATE}\n"
        "AutoTLDR source-flagged documents excluded from clustering fit; eligible rows assigned.\n"
        "No Reddit or hosted LLM API used by this script.\n",
        encoding="utf-8",
    )
    print(f"\nPHASE 7L SEMANTIC MODEL COMPLETE. Outputs saved to: {OUT}")
    print(pd.DataFrame(comparisons).to_string(index=False))

if __name__ == "__main__":
    main()
