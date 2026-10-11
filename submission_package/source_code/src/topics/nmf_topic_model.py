from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# Phase 7A: reproducible lightweight topic-modeling baseline using TF-IDF + NMF.
# Run from the project root:
# python .\src\topics\nmf_topic_model.py

INPUT_FILE = Path("data/processed/reddit_nlp_corpus.csv")
OUTPUT_DIR = Path("results/tables")
N_TOPICS = 8
TOP_N_TERMS = 15
REPRESENTATIVE_DOCS_PER_TOPIC = 8
RANDOM_STATE = 42

# Remove common normalization artifacts seen in the existing corpus.
CUSTOM_STOP_WORDS = {
    "nt", "ve", "gt", "http", "https", "www", "com", "amp",
    "reddit", "subreddit", "deleted", "removed",
    "don", "doesn", "didn", "isn", "wasn", "weren", "wouldn",
    "couldn", "shouldn", "hasn", "haven", "hadn", "won",
    "ll", "re", "s", "t", "d", "m", "ve",
}


def clean_for_vectorizer(value):
    """Light cleanup only; preserve the original text in output files."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    # analysis_text is already normalized; this removes stray punctuation artifacts
    # without changing the saved source text.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("=" * 72)
    print("PHASE 7A — TF-IDF + NMF TOPIC-MODELING BASELINE")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input corpus not found: {INPUT_FILE}\n"
            "Run this script from the project root and check the file path."
        )

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    required = ["record_id", "community", "analysis_text"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"Input corpus is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if df["record_id"].duplicated().any():
        duplicates = int(df["record_id"].duplicated().sum())
        raise ValueError(
            f"Found {duplicates} duplicate record_id values in {INPUT_FILE}. "
            "Resolve duplicates before topic modeling."
        )

    df = df.copy()
    df["analysis_text"] = df["analysis_text"].map(clean_for_vectorizer)
    df = df.loc[df["analysis_text"].str.len() > 0].reset_index(drop=True)

    if len(df) <= N_TOPICS:
        raise ValueError(f"Only {len(df)} non-empty documents; cannot fit {N_TOPICS} topics.")

    print(f"Input file: {INPUT_FILE}")
    print(f"Documents used: {len(df):,}")
    print(f"Communities: {', '.join(sorted(df['community'].dropna().astype(str).unique()))}")
    print(f"Number of topics: {N_TOPICS}")

    vectorizer = TfidfVectorizer(
        stop_words=list(CUSTOM_STOP_WORDS),
        max_features=15000,
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        sublinear_tf=True,
        norm="l2",
    )

    try:
        X = vectorizer.fit_transform(df["analysis_text"])
    except ValueError as exc:
        raise ValueError(
            "TF-IDF vocabulary could not be built. Check the corpus text and "
            "vectorizer filtering settings."
        ) from exc

    if X.shape[1] < N_TOPICS:
        raise ValueError(
            f"TF-IDF created only {X.shape[1]} terms, fewer than {N_TOPICS} topics."
        )

    print(f"TF-IDF matrix: {X.shape[0]:,} documents × {X.shape[1]:,} terms")
    print("Fitting NMF model...")

    model = NMF(
        n_components=N_TOPICS,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=500,
        random_state=RANDOM_STATE,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        W = model.fit_transform(X)
    for warning in caught_warnings:
        print(f"WARNING: {warning.message}")

    feature_names = vectorizer.get_feature_names_out()
    H = model.components_
    topic_rows = []
    for topic_idx, weights in enumerate(H):
        top_indices = np.argsort(weights)[::-1][:TOP_N_TERMS]
        for rank, term_idx in enumerate(top_indices, start=1):
            topic_rows.append(
                {
                    "topic_id": topic_idx + 1,
                    "term_rank": rank,
                    "term": feature_names[term_idx],
                    "term_weight": float(weights[term_idx]),
                }
            )
    topic_terms = pd.DataFrame(topic_rows)

    row_sums = W.sum(axis=1, keepdims=True)
    topic_shares = np.divide(
        W, row_sums, out=np.zeros_like(W), where=row_sums != 0
    )
    dominant_zero_based = W.argmax(axis=1)
    doc_out = df.drop(columns=[], errors="ignore").copy()
    doc_out["dominant_topic"] = dominant_zero_based + 1
    doc_out["dominant_topic_weight"] = W.max(axis=1)
    doc_out["dominant_topic_share"] = topic_shares.max(axis=1)
    for topic_idx in range(N_TOPICS):
        doc_out[f"topic_{topic_idx + 1}_weight"] = W[:, topic_idx]
        doc_out[f"topic_{topic_idx + 1}_share"] = topic_shares[:, topic_idx]

    # Community prevalence is the average normalized topic share, not a causal effect.
    community_rows = []
    for community, group in doc_out.groupby("community", dropna=False):
        for topic_idx in range(N_TOPICS):
            community_rows.append(
                {
                    "community": community,
                    "topic_id": topic_idx + 1,
                    "document_count": len(group),
                    "mean_topic_share": float(group[f"topic_{topic_idx + 1}_share"].mean()),
                    "dominant_topic_document_count": int((group["dominant_topic"] == topic_idx + 1).sum()),
                    "dominant_topic_document_pct": float((group["dominant_topic"] == topic_idx + 1).mean() * 100),
                }
            )
    community_summary = pd.DataFrame(community_rows)

    year_summary = pd.DataFrame()
    if "year" in doc_out.columns:
        year_rows = []
        years = pd.to_numeric(doc_out["year"], errors="coerce")
        doc_out["year"] = years
        for year, group in doc_out.loc[years.notna()].groupby("year"):
            for topic_idx in range(N_TOPICS):
                year_rows.append(
                    {
                        "year": int(year),
                        "topic_id": topic_idx + 1,
                        "document_count": len(group),
                        "mean_topic_share": float(group[f"topic_{topic_idx + 1}_share"].mean()),
                        "dominant_topic_document_count": int((group["dominant_topic"] == topic_idx + 1).sum()),
                        "dominant_topic_document_pct": float((group["dominant_topic"] == topic_idx + 1).mean() * 100),
                    }
                )
        year_summary = pd.DataFrame(year_rows)

    representative_rows = []
    for topic_idx in range(N_TOPICS):
        topic_share_col = f"topic_{topic_idx + 1}_share"
        top_doc_indices = np.argsort(topic_shares[:, topic_idx])[::-1][
            :REPRESENTATIVE_DOCS_PER_TOPIC
        ]
        for rank, row_idx in enumerate(top_doc_indices, start=1):
            representative_rows.append(
                {
                    "topic_id": topic_idx + 1,
                    "representative_rank": rank,
                    "topic_share": float(topic_shares[row_idx, topic_idx]),
                    "record_id": df.iloc[row_idx]["record_id"],
                    "community": df.iloc[row_idx]["community"],
                    "year": df.iloc[row_idx]["year"] if "year" in df.columns else np.nan,
                    "text_excerpt": str(df.iloc[row_idx]["analysis_text"])[:500],
                }
            )
    representatives = pd.DataFrame(representative_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "nmf_topic_terms_k8.csv": topic_terms,
        "nmf_document_topics_k8.csv": doc_out,
        "nmf_topic_community_distribution_k8.csv": community_summary,
        "nmf_representative_documents_k8.csv": representatives,
    }
    if not year_summary.empty:
        outputs["nmf_topic_year_distribution_k8.csv"] = year_summary

    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False)
        print(f"Saved: {path}")

    diagnostics_path = OUTPUT_DIR / "nmf_diagnostics_k8.txt"
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        handle.write("PHASE 7A — TF-IDF + NMF TOPIC-MODELING BASELINE\n")
        handle.write(f"Input file: {INPUT_FILE}\n")
        handle.write(f"Documents used: {X.shape[0]}\n")
        handle.write(f"Vocabulary size: {X.shape[1]}\n")
        handle.write(f"Number of topics: {N_TOPICS}\n")
        handle.write(f"Reconstruction error: {model.reconstruction_err_:.8f}\n")
        handle.write(f"Iterations: {model.n_iter_}\n")
        handle.write(f"Random state: {RANDOM_STATE}\n\n")
        handle.write("Top terms by topic:\n")
        for topic_idx in range(N_TOPICS):
            terms = topic_terms.loc[
                topic_terms["topic_id"] == topic_idx + 1, "term"
            ].tolist()
            handle.write(f"Topic {topic_idx + 1}: {', '.join(terms)}\n")
        handle.write(
            "\nReminder: topics are exploratory clusters of word co-occurrence, "
            "not validated categories or psychological traits. Review representative "
            "documents before assigning human-readable topic labels.\n"
        )
    print(f"Saved: {diagnostics_path}")

    print("\nTop terms by topic:")
    for topic_idx in range(N_TOPICS):
        terms = topic_terms.loc[
            topic_terms["topic_id"] == topic_idx + 1, "term"
        ].tolist()
        print(f"Topic {topic_idx + 1}: {', '.join(terms)}")

    print("\nModel diagnostics:")
    print(f"Reconstruction error: {model.reconstruction_err_:.6f}")
    print(f"Iterations: {model.n_iter_}")
    print("\nPhase 7A complete.")
    print("Next: inspect top terms and representative documents before naming topics.")


if __name__ == "__main__":
    main()
