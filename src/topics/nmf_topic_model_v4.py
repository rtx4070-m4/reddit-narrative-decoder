"""
Phase 7D: focused NMF topic-model preprocessing experiment.

Reads the stable Phase 7B document-topic file for its document IDs, community,
date, and analysis text. Creates a topic-model-only text representation; it
does not alter the source corpus or earlier outputs.

Run from the project root:
    python src/topics/nmf_topic_model_v4.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables" / "nmf_document_topics_k8_v2.csv"
OUT = ROOT / "results" / "tables" / "phase7d_nmf"
OUT.mkdir(parents=True, exist_ok=True)

K = 8
MIN_CLEAN_TOKENS = 6
RANDOM_STATE = 42
TOP_TERMS = 15
TOP_DOCS_PER_TOPIC = 8

# Conservative additions: suppress common conversational scaffolding, but
# keep topic-bearing nouns and verbs wherever possible.
EXTRA_STOPWORDS = {
    "just", "like", "does", "did", "really", "thing", "things", "people",
    "know", "think", "want", "going", "got", "get", "make", "way",
    "good", "great", "nice", "right", "time", "look", "looks", "seems",
    "maybe", "actually", "pretty", "quite", "kind", "sort", "lot",
    "say", "said", "says", "come", "came", "use", "used", "using",
    "need", "sure", "yeah", "yes", "okay", "ok", "well", "im", "ive",
    "youre", "thats", "dont", "doesnt", "didnt", "cant", "couldnt",
    "would", "could", "should", "also", "still", "much", "many",
    "something", "anything", "someone", "anyone", "one", "two",
    "http", "https", "www", "com", "reddit", "comments", "npredditcom",
    "autotldr", "feedback", "quot", "amp", "gt", "quotgt", "tl",
    "tldr", "bot", "original", "version", "welcome", "reduced",
}
STOPWORDS = sorted(set(ENGLISH_STOP_WORDS).union(EXTRA_STOPWORDS))

def clean_for_topics(value):
    """Topic-model-only cleaner; leaves original text untouched."""
    text = "" if pd.isna(value) else str(value).lower()
    # Remove recognizable bot-summary boilerplate before tokenization.
    patterns = [
        r"this is the best tl dr i could make",
        r"i am a bot",
        r"i m a bot",
        r"reduced by \d+ percent",
        r"feedback welcome",
        r"original post",
        r"original article",
        r"npredditcomrautotldrcomments",
        r"wwwyoutubecomwatch",
        r"https?\S+",
        r"\b(?:quot|amp|gt)\b",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text)
    # Keep alphabetic terms only; avoids malformed URL and punctuation artifacts.
    tokens = re.findall(r"[a-z]{2,}", text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Required input not found: {INPUT}\n"
            "Check that Phase 7B output nmf_document_topics_k8_v2.csv exists."
        )

    df = pd.read_csv(INPUT)
    required = {"analysis_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    # Preserve source rows and text; this column is only for this experiment.
    df["topic_clean_text"] = df["analysis_text"].map(clean_for_topics)
    df["topic_clean_token_count"] = df["topic_clean_text"].str.split().str.len()
    eligible = df[df["topic_clean_token_count"] >= MIN_CLEAN_TOKENS].copy()
    excluded_short = df[df["topic_clean_token_count"] < MIN_CLEAN_TOKENS].copy()

    if len(eligible) < K:
        raise ValueError(f"Only {len(eligible)} eligible documents; cannot fit K={K}.")

    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        min_df=8,
        max_df=0.70,
        max_features=12000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z]{2,}\b",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(eligible["topic_clean_text"])
    vocab = np.asarray(vectorizer.get_feature_names_out())

    # Documents that pass token length but contain no retained vocabulary are
    # exported as unmodelled, rather than silently assigned a topic.
    row_has_features = np.asarray(X.getnnz(axis=1)).ravel() > 0
    model_df = eligible.loc[row_has_features].copy().reset_index(drop=True)
    X_model = X[row_has_features]
    unmodelled = eligible.loc[~row_has_features].copy()
    if X_model.shape[0] < K or X_model.shape[1] < K:
        raise ValueError(
            f"Insufficient usable matrix for K={K}: {X_model.shape}. "
            "Relax min_df or the cleaning rules."
        )

    model = NMF(
        n_components=K,
        init="nndsvda",
        random_state=RANDOM_STATE,
        max_iter=400,
        tol=1e-4,
    )
    W = model.fit_transform(X_model)
    H = model.components_
    sums = W.sum(axis=1, keepdims=True)
    shares = np.divide(W, sums, out=np.zeros_like(W), where=sums > 0)
    dominant = shares.argmax(axis=1)
    max_share = shares.max(axis=1)

    # Metadata column names are taken from the input when available.
    metadata_candidates = [
        "record_id", "id", "community", "subreddit", "created_utc",
        "created_at", "year", "text", "analysis_text"
    ]
    meta_cols = [c for c in metadata_candidates if c in model_df.columns]
    doc_out = model_df[meta_cols].copy()
    doc_out["topic_clean_token_count"] = model_df["topic_clean_token_count"]
    for i in range(K):
        doc_out[f"topic_{i+1}_weight"] = W[:, i]
        doc_out[f"topic_{i+1}_share"] = shares[:, i]
    doc_out["dominant_topic"] = dominant + 1
    doc_out["dominant_topic_share"] = max_share
    doc_out.to_csv(OUT / f"phase7d_document_topics_k{K}.csv", index=False)

    terms_rows = []
    for topic_idx, weights in enumerate(H):
        top_idx = np.argsort(weights)[::-1][:TOP_TERMS]
        for rank, term_idx in enumerate(top_idx, start=1):
            terms_rows.append({
                "topic": topic_idx + 1,
                "rank": rank,
                "term": vocab[term_idx],
                "term_weight": float(weights[term_idx]),
            })
    pd.DataFrame(terms_rows).to_csv(
        OUT / f"phase7d_topic_terms_k{K}.csv", index=False
    )

    # Representatives are ranked by normalized topic share, with token count
    # as a tie-breaker; show multiple candidates for manual interpretability.
    rep_rows = []
    for topic_idx in range(K):
        ranked = doc_out.copy()
        ranked["candidate_share"] = shares[:, topic_idx]
        ranked = ranked.sort_values(
            ["candidate_share", "topic_clean_token_count"],
            ascending=[False, False],
            kind="mergesort",
        ).head(TOP_DOCS_PER_TOPIC)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rep = {
                "topic": topic_idx + 1,
                "rank": rank,
                "topic_share": row["candidate_share"],
                "topic_clean_token_count": row["topic_clean_token_count"],
            }
            for col in ["record_id", "id", "community", "subreddit", "year", "analysis_text"]:
                if col in row.index:
                    rep[col] = row[col]
            rep_rows.append(rep)
    pd.DataFrame(rep_rows).to_csv(
        OUT / f"phase7d_representative_documents_k{K}.csv", index=False
    )

    community_col = next(
        (c for c in ["community", "subreddit"] if c in doc_out.columns), None
    )
    if community_col:
        dist = (
            doc_out.groupby([community_col, "dominant_topic"])
            .size().rename("document_count").reset_index()
        )
        totals = dist.groupby(community_col)["document_count"].transform("sum")
        dist["within_community_share"] = dist["document_count"] / totals
        dist.to_csv(OUT / f"phase7d_community_distribution_k{K}.csv", index=False)

    if len(excluded_short):
        excluded_short.to_csv(OUT / "phase7d_excluded_short_documents.csv", index=False)
    if len(unmodelled):
        unmodelled.to_csv(OUT / "phase7d_unmodelled_documents.csv", index=False)

    diagnostics = [
        "PHASE 7D NMF — PREPROCESSING EXPERIMENT",
        f"Input: {INPUT}",
        f"Input rows: {len(df)}",
        f"Eligible after topic-only cleaning (>= {MIN_CLEAN_TOKENS} tokens): {len(eligible)}",
        f"Excluded for short cleaned text: {len(excluded_short)}",
        f"Eligible but no retained TF-IDF features: {len(unmodelled)}",
        f"Documents fitted: {X_model.shape[0]}",
        f"Vocabulary size: {X_model.shape[1]}",
        f"K: {K}",
        f"Iterations: {model.n_iter_}",
        f"Reconstruction error: {model.reconstruction_err_:.6f}",
        f"Random state: {RANDOM_STATE}",
        "",
        "Interpretation cautions:",
        "- Topic-model cleaning is experimental and does not overwrite original text.",
        "- Generic-word stopword additions may remove meaningful content; inspect topic terms and representatives.",
        "- NMF topics are lexical mixtures, not validated semantic categories.",
        "- Compare topic coherence/readability and representative documents against Phase 7C; do not select solely by reconstruction error.",
        "- Excluded documents remain documented in output files and should not be treated as absent from the source corpus.",
    ]
    (OUT / "phase7d_diagnostics.txt").write_text("\n".join(diagnostics), encoding="utf-8")

    print("\n".join(diagnostics))
    print(f"\nOutputs saved to: {OUT}")

if __name__ == "__main__":
    main()
