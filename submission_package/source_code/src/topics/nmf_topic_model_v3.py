"""
PHASE 7C — TOPIC MODEL COMPARISON (NMF)
=======================================

Purpose
-------
Compare several NMF topic counts on the same corpus and topic-specific text
representation. This is exploratory model selection, not a guarantee that the
lowest reconstruction error or highest topic diversity is the best model.

Input
-----
results/tables/nmf_document_topics_k8_v2.csv

This preserves the exact 4,852-document Phase 7B analysis set and its metadata.
The existing Phase 7B outputs are not modified.

Outputs
-------
results/tables/phase7c_nmf/
  phase7c_model_comparison.csv
  phase7c_topic_terms_k{K}.csv
  phase7c_document_topics_k{K}.csv
  phase7c_representatives_k{K}.csv
  phase7c_community_distribution_k{K}.csv
  phase7c_unmodelled_documents.csv
  phase7c_run_notes.txt

Run from the project root:
  python .\\src\\topics\\nmf_topic_model_v3.py
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

INPUT_FILE = Path("results/tables/nmf_document_topics_k8_v2.csv")
OUTPUT_DIR = Path("results/tables/phase7c_nmf")

TOPIC_COUNTS = [4, 5, 6, 8, 10, 12]
TOP_N_TERMS = 15
REPRESENTATIVES_PER_TOPIC = 8
RANDOM_STATE = 42
MAX_ITER = 500

# Topic-model-specific filtering only. The source text and prior NLP outputs
# are retained unchanged. Do not use this transformed text for emotion analysis.
CUSTOM_STOP_WORDS = {
    # URL / markup artifacts
    "http", "https", "www", "com", "amp", "html", "jpg", "jpeg", "png",
    "gif", "watch", "youtube", "youtu", "youtu be",
    # Contractions split by prior text normalization
    "nt", "ve", "ll", "re", "d", "ca", "im", "ive", "dont", "doesnt",
    # Generic conversational fillers observed in Phase 7B
    "actually", "really", "thing", "things", "way", "ways", "yeah",
    "yep", "okay", "ok", "oh", "wow", "right", "got", "going",
    "know", "think", "mean", "said", "says", "say", "make",
    # Generic terms that dominated the earlier model
    "people", "time", "good", "great",
}

STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS).union(CUSTOM_STOP_WORDS))


def clean_for_topics(text):
    """Clean URL remnants and ID-like artifacts for topic modeling only."""
    text = "" if pd.isna(text) else str(text).lower()

    # Remove URL-like sequences before tokenization.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove normalized URL residue such as wwwyoutubecomwatch and
    # common domain/path fragments when punctuation has already disappeared.
    text = re.sub(
        r"\b(?:www)?(?:youtube|youtu|imgur|reddit|twitter|facebook|"
        r"instagram|twitch|steamcommunity|steampowered|comwatch|watchv)\w*\b",
        " ",
        text,
    )
    # Remove long alphanumeric IDs / hash-like strings.
    text = re.sub(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]{8,}\b", " ", text)
    # Keep alphabetic words; TF-IDF will use its token pattern as a second filter.
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def top_term_diversity(topic_terms, top_n=10):
    """Unique top terms divided by all top-term slots; descriptive only."""
    top = topic_terms[topic_terms["term_rank"] <= top_n]["term"].tolist()
    slots = len(top)
    return (len(set(top)) / slots) if slots else np.nan


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}\n"
            "Run this script from the project root and confirm Phase 7B completed."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    text_col = "analysis_text" if "analysis_text" in df.columns else "text"
    required = {"record_id", "community", text_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["topic_text_v3"] = df[text_col].map(clean_for_topics)

    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        max_features=12000,
        min_df=8,
        max_df=0.70,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z]{2,}\b",
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X_all = vectorizer.fit_transform(df["topic_text_v3"])
    row_nonzero = np.asarray(X_all.getnnz(axis=1)).ravel() > 0
    model_df = df.loc[row_nonzero].copy().reset_index(drop=True)
    X = X_all[row_nonzero]

    unmodelled = df.loc[~row_nonzero].copy()
    keep_cols = [
        c for c in ["record_id", "community", "year", "analysis_text", "text"]
        if c in unmodelled.columns
    ]
    unmodelled[keep_cols].to_csv(
        OUTPUT_DIR / "phase7c_unmodelled_documents.csv", index=False
    )

    if X.shape[0] < max(TOPIC_COUNTS):
        raise ValueError(
            f"Only {X.shape[0]} documents have usable features; "
            f"cannot test K={max(TOPIC_COUNTS)}."
        )
    if X.shape[1] < max(TOPIC_COUNTS):
        raise ValueError(
            f"Only {X.shape[1]} vocabulary terms remain; "
            f"cannot test K={max(TOPIC_COUNTS)}."
        )

    terms = vectorizer.get_feature_names_out()
    print("=" * 72)
    print("PHASE 7C — COMPARE NMF TOPIC COUNTS")
    print("=" * 72)
    print(f"Input documents: {len(df):,}")
    print(f"Documents with usable TF-IDF features: {X.shape[0]:,}")
    print(f"Documents with no usable features: {int((~row_nonzero).sum()):,}")
    print(f"TF-IDF vocabulary: {X.shape[1]:,}")
    print(f"Candidate topic counts: {TOPIC_COUNTS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Fitting models; this may take a few minutes...")

    comparison_rows = []
    notes = [
        "PHASE 7C NMF MODEL COMPARISON",
        f"Input file: {INPUT_FILE}",
        f"Input documents: {len(df)}",
        f"Modelled documents: {X.shape[0]}",
        f"Unmodelled documents: {int((~row_nonzero).sum())}",
        f"TF-IDF vocabulary size: {X.shape[1]}",
        f"Candidate K values: {TOPIC_COUNTS}",
        "",
        "Interpretation cautions:",
        "- Reconstruction error generally changes with topic count; do not select K from it alone.",
        "- Top-term diversity is a simple descriptive metric, not a validated coherence score.",
        "- Inspect representative documents and community distributions before naming topics.",
        "- Topic labels describe corpus-level text patterns, not individual users or mental states.",
        "- The topic-specific text is only for this experiment; original text remains unchanged.",
    ]

    for k in TOPIC_COUNTS:
        print(f"\nFitting NMF with K={k}...")
        model = NMF(
            n_components=k,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            W = model.fit_transform(X)
        for warning in caught:
            print(f"WARNING (K={k}): {warning.message}")

        row_sums = W.sum(axis=1, keepdims=True)
        shares = np.divide(
            W, row_sums, out=np.zeros_like(W), where=row_sums > 1e-12
        )
        dominant = W.argmax(axis=1) + 1
        dominant_share = shares.max(axis=1)

        term_rows = []
        for topic_idx, component in enumerate(model.components_):
            top_indices = np.argsort(component)[::-1][:TOP_N_TERMS]
            for rank, term_idx in enumerate(top_indices, start=1):
                term_rows.append({
                    "topic_id": topic_idx + 1,
                    "term_rank": rank,
                    "term": terms[term_idx],
                    "term_weight": float(component[term_idx]),
                })
        topic_terms = pd.DataFrame(term_rows)
        topic_terms.to_csv(
            OUTPUT_DIR / f"phase7c_topic_terms_k{k}.csv", index=False
        )

        # Per-document topic weights and normalized shares.
        doc_out = model_df.copy()
        doc_out["dominant_topic"] = dominant
        doc_out["dominant_topic_weight"] = W.max(axis=1)
        doc_out["dominant_topic_share"] = dominant_share
        for topic_idx in range(k):
            doc_out[f"topic_{topic_idx + 1}_weight"] = W[:, topic_idx]
            doc_out[f"topic_{topic_idx + 1}_share"] = shares[:, topic_idx]
        doc_out = doc_out.drop(columns=["topic_text_v3"], errors="ignore")
        doc_out.to_csv(
            OUTPUT_DIR / f"phase7c_document_topics_k{k}.csv", index=False
        )

        # Representative docs: exclude very short documents where possible,
        # and rank by raw topic weight. Fall back to all modelled docs if needed.
        reps = []
        word_counts = model_df["topic_text_v3"].str.split().str.len().fillna(0)
        for topic_idx in range(k):
            topic_weight = W[:, topic_idx]
            candidate_mask = (topic_weight > 0) & (word_counts >= 5)
            if candidate_mask.sum() < REPRESENTATIVES_PER_TOPIC:
                candidate_mask = topic_weight > 0
            candidate_indices = np.flatnonzero(candidate_mask)
            ordered = candidate_indices[
                np.argsort(topic_weight[candidate_indices])[::-1]
            ][:REPRESENTATIVES_PER_TOPIC]

            for rank, idx in enumerate(ordered, start=1):
                source_text = model_df.iloc[idx][text_col]
                reps.append({
                    "topic_id": topic_idx + 1,
                    "representative_rank": rank,
                    "topic_weight": float(W[idx, topic_idx]),
                    "topic_share": float(shares[idx, topic_idx]),
                    "record_id": model_df.iloc[idx]["record_id"],
                    "community": model_df.iloc[idx]["community"],
                    "year": model_df.iloc[idx].get("year", np.nan),
                    "text_excerpt": str(source_text)[:500],
                    "topic_model_text": model_df.iloc[idx]["topic_text_v3"][:500],
                    "topic_word_count": int(word_counts.iloc[idx]),
                })
        pd.DataFrame(reps).to_csv(
            OUTPUT_DIR / f"phase7c_representatives_k{k}.csv", index=False
        )

        # Community distribution uses only modelled documents. Document this
        # denominator in output; unmodelled documents are reported separately.
        community_rows = []
        temp = pd.DataFrame({
            "community": model_df["community"].to_numpy(),
            "dominant_topic": dominant,
        })
        for community, group in temp.groupby("community", dropna=False):
            mask = temp["community"].eq(community).to_numpy()
            for topic_idx in range(k):
                community_rows.append({
                    "community": community,
                    "topic_id": topic_idx + 1,
                    "modelled_document_count": int(mask.sum()),
                    "dominant_topic_document_count": int(
                        np.sum(mask & (dominant == topic_idx + 1))
                    ),
                    "dominant_topic_document_pct": float(
                        np.mean(dominant[mask] == topic_idx + 1) * 100
                    ),
                    "mean_topic_share": float(shares[mask, topic_idx].mean()),
                })
        pd.DataFrame(community_rows).to_csv(
            OUTPUT_DIR / f"phase7c_community_distribution_k{k}.csv",
            index=False,
        )

        diversity = top_term_diversity(topic_terms, top_n=10)
        topic_counts = np.bincount(dominant, minlength=k + 1)[1:]
        proportions = topic_counts / len(dominant)
        nonzero_p = proportions[proportions > 0]
        entropy = float(-(nonzero_p * np.log(nonzero_p)).sum())
        normalized_entropy = entropy / np.log(k) if k > 1 else 0.0

        comparison_rows.append({
            "n_topics": k,
            "n_documents_modelled": int(X.shape[0]),
            "vocabulary_size": int(X.shape[1]),
            "reconstruction_error": float(model.reconstruction_err_),
            "iterations": int(model.n_iter_),
            "top10_term_diversity": diversity,
            "dominant_topic_entropy": entropy,
            "normalized_dominant_topic_entropy": normalized_entropy,
            "smallest_dominant_topic_pct": float(proportions.min() * 100),
            "largest_dominant_topic_pct": float(proportions.max() * 100),
            "documents_with_dominant_share_ge_0_80_pct": float(
                np.mean(dominant_share >= 0.80) * 100
            ),
        })
        print(
            f"K={k}: reconstruction_error={model.reconstruction_err_:.4f}; "
            f"iterations={model.n_iter_}; top10_diversity={diversity:.3f}; "
            f"largest_topic={proportions.max() * 100:.1f}%"
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "phase7c_model_comparison.csv", index=False)
    (OUTPUT_DIR / "phase7c_run_notes.txt").write_text(
        "\n".join(notes) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 72)
    print(comparison.round(4).to_string(index=False))
    print("\nPhase 7C complete.")
    print(f"Comparison: {OUTPUT_DIR / 'phase7c_model_comparison.csv'}")
    print("Next: inspect K=5, K=8, and whichever K appears most balanced.")
    print("Do not choose K from reconstruction error alone.")


if __name__ == "__main__":
    main()
