from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

INPUT_FILE = Path("data/processed/reddit_nlp_corpus.csv")
OUTPUT_DIR = Path("results/tables")
N_TOPICS = 8
TOP_N_TERMS = 15
REPRESENTATIVE_DOCS_PER_TOPIC = 8
RANDOM_STATE = 42

# Remove common artifacts from the corpus's normalized URL/text representation.
ARTIFACT_STOP_WORDS = {
    "nt", "ve", "gt", "http", "https", "www", "com", "amp",
    "reddit", "subreddit", "deleted", "removed",
    "ll", "re", "s", "t", "d", "m", "ve",
    "youtube", "watch", "imgur", "jpg", "png", "gif",
}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS).union(ARTIFACT_STOP_WORDS))


def clean_for_topics(value):
    """Conservative cleanup for topic modeling; never overwrite source text."""
    if pd.isna(value):
        return ""
    text = str(value).lower()
    # Remove URL strings and common normalized URL remnants.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b(?:https?|www|com|youtube|watch|imgur|jpg|jpeg|png|gif)\b", " ", text)
    # Remove long URL-like tokens / identifiers, retaining ordinary words.
    text = re.sub(r"\b[a-z0-9]*\d[a-z0-9]{5,}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("=" * 72)
    print("PHASE 7B — IMPROVED TF-IDF + NMF TOPIC MODEL")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input corpus not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    required = ["record_id", "community", "analysis_text"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input corpus missing columns: {missing}")

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id values found; resolve before modeling.")

    df = df.copy()
    df["topic_text"] = df["analysis_text"].map(clean_for_topics)
    df = df.loc[df["topic_text"].str.len() > 0].reset_index(drop=True)

    print(f"Documents used: {len(df):,}")
    print(f"Communities: {', '.join(sorted(df['community'].dropna().astype(str).unique()))}")
    print(f"Stopword count: {len(STOP_WORDS):,}")
    print(f"Topics: {N_TOPICS}")

    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS,
        max_features=12000,
        min_df=8,
        max_df=0.65,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        sublinear_tf=True,
        norm="l2",
    )

    try:
        X = vectorizer.fit_transform(df["topic_text"])
    except ValueError as exc:
        raise ValueError(
            "Could not build a TF-IDF vocabulary. Inspect text cleaning and min_df/max_df."
        ) from exc

    if X.shape[1] < N_TOPICS:
        raise ValueError(f"Vocabulary too small for NMF: {X.shape[1]} terms.")

    print(f"TF-IDF matrix: {X.shape[0]:,} documents × {X.shape[1]:,} terms")
    print("Fitting NMF...")
    model = NMF(
        n_components=N_TOPICS,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=600,
        random_state=RANDOM_STATE,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        W = model.fit_transform(X)
    for warning in caught:
        print(f"WARNING: {warning.message}")

    terms = vectorizer.get_feature_names_out()
    H = model.components_
    term_rows = []
    for k, weights in enumerate(H):
        indices = np.argsort(weights)[::-1][:TOP_N_TERMS]
        for rank, term_idx in enumerate(indices, start=1):
            term_rows.append({
                "topic_id": k + 1,
                "term_rank": rank,
                "term": terms[term_idx],
                "term_weight": float(weights[term_idx]),
            })
    topic_terms = pd.DataFrame(term_rows)

    sums = W.sum(axis=1, keepdims=True)
    shares = np.divide(W, sums, out=np.zeros_like(W), where=sums != 0)

    document_topics = df.drop(columns=["topic_text"]).copy()
    document_topics["dominant_topic"] = W.argmax(axis=1) + 1
    document_topics["dominant_topic_weight"] = W.max(axis=1)
    document_topics["dominant_topic_share"] = shares.max(axis=1)
    for k in range(N_TOPICS):
        document_topics[f"topic_{k + 1}_weight"] = W[:, k]
        document_topics[f"topic_{k + 1}_share"] = shares[:, k]

    community_rows = []
    for community, group in document_topics.groupby("community", dropna=False):
        for k in range(N_TOPICS):
            share_col = f"topic_{k + 1}_share"
            dominant = group["dominant_topic"] == k + 1
            community_rows.append({
                "community": community,
                "topic_id": k + 1,
                "document_count": len(group),
                "mean_topic_share": float(group[share_col].mean()),
                "dominant_topic_document_count": int(dominant.sum()),
                "dominant_topic_document_pct": float(dominant.mean() * 100),
            })
    community_summary = pd.DataFrame(community_rows)

    year_summary = pd.DataFrame()
    if "year" in document_topics.columns:
        years = pd.to_numeric(document_topics["year"], errors="coerce")
        year_rows = []
        for year, group in document_topics.loc[years.notna()].groupby(years[years.notna()]):
            for k in range(N_TOPICS):
                dominant = group["dominant_topic"] == k + 1
                year_rows.append({
                    "year": int(year),
                    "topic_id": k + 1,
                    "document_count": len(group),
                    "mean_topic_share": float(group[f"topic_{k + 1}_share"].mean()),
                    "dominant_topic_document_count": int(dominant.sum()),
                    "dominant_topic_document_pct": float(dominant.mean() * 100),
                })
        year_summary = pd.DataFrame(year_rows)

    representative_rows = []
    for k in range(N_TOPICS):
        top_indices = np.argsort(shares[:, k])[::-1][:REPRESENTATIVE_DOCS_PER_TOPIC]
        for rank, row_idx in enumerate(top_indices, start=1):
            representative_rows.append({
                "topic_id": k + 1,
                "representative_rank": rank,
                "topic_share": float(shares[row_idx, k]),
                "record_id": df.iloc[row_idx]["record_id"],
                "community": df.iloc[row_idx]["community"],
                "year": df.iloc[row_idx]["year"] if "year" in df.columns else np.nan,
                "text_excerpt": str(df.iloc[row_idx]["analysis_text"])[:500],
            })
    representatives = pd.DataFrame(representative_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "nmf_topic_terms_k8_v2.csv": topic_terms,
        "nmf_document_topics_k8_v2.csv": document_topics,
        "nmf_topic_community_distribution_k8_v2.csv": community_summary,
        "nmf_representative_documents_k8_v2.csv": representatives,
    }
    if not year_summary.empty:
        outputs["nmf_topic_year_distribution_k8_v2.csv"] = year_summary

    for filename, frame in outputs.items():
        out_path = OUTPUT_DIR / filename
        frame.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    diagnostics_path = OUTPUT_DIR / "nmf_diagnostics_k8_v2.txt"
    with diagnostics_path.open("w", encoding="utf-8") as f:
        f.write("PHASE 7B — IMPROVED TF-IDF + NMF\n")
        f.write(f"Documents: {X.shape[0]}\nVocabulary size: {X.shape[1]}\n")
        f.write(f"Topics: {N_TOPICS}\nRandom state: {RANDOM_STATE}\n")
        f.write(f"Reconstruction error: {model.reconstruction_err_:.8f}\n")
        f.write(f"Iterations: {model.n_iter_}\n\n")
        for k in range(N_TOPICS):
            top_terms = topic_terms.loc[topic_terms["topic_id"] == k + 1, "term"].tolist()
            f.write(f"Topic {k + 1}: {', '.join(top_terms)}\n")
        f.write(
            "\nTopics are exploratory word-co-occurrence patterns. Inspect representative "
            "documents and compare model quality before assigning final labels.\n"
        )
    print(f"Saved: {diagnostics_path}")

    print("\nTop terms by topic:")
    for k in range(N_TOPICS):
        top_terms = topic_terms.loc[topic_terms["topic_id"] == k + 1, "term"].tolist()
        print(f"Topic {k + 1}: {', '.join(top_terms)}")
    print("\nModel diagnostics:")
    print(f"Reconstruction error: {model.reconstruction_err_:.6f}")
    print(f"Iterations: {model.n_iter_}")
    print("\nPhase 7B complete. Review terms and representative documents before labeling topics.")


if __name__ == "__main__":
    main()
