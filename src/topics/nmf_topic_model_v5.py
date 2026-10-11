"""
Phase 7E: NMF topic model using original master-corpus text.

Purpose:
- Repair common contractions before tokenization.
- Remove recognizable AutoTLDR boilerplate from topic-model input only.
- Use a lower minimum cleaned-token threshold than Phase 7D.
- Fit K=8 for a controlled preprocessing comparison.
- Preserve earlier phases and export exclusions/diagnostics separately.

Run from the project root:
    python src/topics/nmf_topic_model_v5.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[2]
MASTER_PATH = ROOT / "data" / "processed" / "reddit_master.csv"
PHASE7B_PATH = ROOT / "results" / "tables" / "nmf_document_topics_k8_v2.csv"
OUT = ROOT / "results" / "tables" / "phase7e_nmf"
OUT.mkdir(parents=True, exist_ok=True)

K = 8
MIN_CLEAN_TOKENS = 3
RANDOM_STATE = 42
TOP_TERMS = 15
TOP_DOCS_PER_TOPIC = 8

# Do not aggressively remove substantive content terms. These are mostly
# obvious corpus/URL artifacts and conversational fillers from earlier runs.
EXTRA_STOPWORDS = {
    "http", "https", "www", "com", "reddit", "comments", "npredditcom",
    "autotldr", "tldr", "quot", "amp", "gt", "quotgt", "feedback",
    "bot", "welcome", "version", "reduced",
    "just", "really", "actually", "pretty", "quite", "maybe", "thing",
    "things", "something", "anything", "someone", "anyone", "lot",
    "yeah", "okay", "ok", "well", "got", "get", "going",
}
STOPWORDS = sorted(set(ENGLISH_STOP_WORDS).union(EXTRA_STOPWORDS))

# Expand contractions before tokenization. Do this only in the derived
# topic-model text, never in the source/master corpus.
CONTRACTIONS = [
    (r"\bcan't\b", "cannot"),
    (r"\bcannot\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bshan't\b", "shall not"),
    (r"\bain't\b", "is not"),
    (r"\bdon't\b", "do not"),
    (r"\bdoesn't\b", "does not"),
    (r"\bdidn't\b", "did not"),
    (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"),
    (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"),
    (r"\bhaven't\b", "have not"),
    (r"\bhasn't\b", "has not"),
    (r"\bhadn't\b", "had not"),
    (r"\bcouldn't\b", "could not"),
    (r"\bwouldn't\b", "would not"),
    (r"\bshouldn't\b", "should not"),
    (r"\bmustn't\b", "must not"),
    (r"\bi'm\b", "i am"),
    (r"\bi've\b", "i have"),
    (r"\bi'll\b", "i will"),
    (r"\bi'd\b", "i would"),
    (r"\byou're\b", "you are"),
    (r"\byou've\b", "you have"),
    (r"\byou'll\b", "you will"),
    (r"\byou'd\b", "you would"),
    (r"\bwe're\b", "we are"),
    (r"\bwe've\b", "we have"),
    (r"\bwe'll\b", "we will"),
    (r"\bwe'd\b", "we would"),
    (r"\bthey're\b", "they are"),
    (r"\bthey've\b", "they have"),
    (r"\bthey'll\b", "they will"),
    (r"\bthey'd\b", "they would"),
    (r"\bit's\b", "it is"),
    (r"\bthat's\b", "that is"),
    (r"\bthere's\b", "there is"),
    (r"\bwhat's\b", "what is"),
    (r"\bwho's\b", "who is"),
    (r"\bhere's\b", "here is"),
    (r"\blet's\b", "let us"),
]

def clean_original_text(value):
    text = "" if pd.isna(value) else str(value).lower()

    # Remove identifiable bot-summary boilerplate, preserving the article
    # summary/content after the metadata marker where possible.
    text = re.sub(r"this is the best tl\s*;?\s*dr i could make", " ", text)
    text = re.sub(r"this is the best tl dr i could make", " ", text)
    text = re.sub(r"\bi am a bot\b|\bi'm a bot\b", " ", text)
    text = re.sub(r"\breduced by\s+\d+\s*(?:percent|%)?", " ", text)
    text = re.sub(r"\bfeedback welcome\b", " ", text)
    text = re.sub(r"\boriginal\s+(?:post|article)\b", " ", text)

    # Remove URLs before deleting punctuation, so domain strings do not become
    # artificial vocabulary terms. Also remove common Reddit AutoTLDR link text.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b(?:npredditcomrautotldrcomments|autotldr)\b", " ", text)

    # Expand common contractions before apostrophes are stripped.
    for pattern, replacement in CONTRACTIONS:
        text = re.sub(pattern, replacement, text)

    # Normalize HTML entities and common flattened artifacts.
    text = re.sub(r"&(?:amp|quot|#\d+|#x[0-9a-f]+);?", " ", text)
    text = re.sub(r"\b(?:amp|quot|gt)\b", " ", text)

    tokens = re.findall(r"[a-z]{2,}", text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def find_column(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of required candidate columns found: {candidates}")
    return None

def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Master corpus not found: {MASTER_PATH}")
    if not PHASE7B_PATH.exists():
        raise FileNotFoundError(f"Phase 7B reference file not found: {PHASE7B_PATH}")

    master = pd.read_csv(MASTER_PATH, low_memory=False)
    reference = pd.read_csv(PHASE7B_PATH, low_memory=False)

    master_id = find_column(master, ["record_id"], required=True)
    master_text = find_column(master, ["text"], required=True)
    master_community = find_column(master, ["community", "subreddit"])
    master_date = find_column(master, ["date", "created_utc"])
    master_year = find_column(master, ["year"])

    ref_id = find_column(reference, ["record_id"], required=True)
    ref_text = find_column(reference, ["analysis_text", "text"], required=True)
    ref_community = find_column(reference, ["community", "subreddit"])
    ref_date = find_column(reference, ["date", "created_utc"])
    ref_year = find_column(reference, ["year"])

    # The Phase 7B reference file defines the stable working set (4,852 rows).
    # Resolve master rows by record ID and community; do not merge by Reddit
    # comment ID alone. If the master has duplicate copies, prefer the longest
    # non-null original text within that exact record_id/community pair.
    key_cols = [master_id]
    ref_key_cols = [ref_id]
    if master_community and ref_community:
        key_cols.append(master_community)
        ref_key_cols.append(ref_community)

    master[master_text] = master[master_text].fillna("").astype(str)
    master["_original_length"] = master[master_text].str.len()
    master_sorted = master.sort_values("_original_length", ascending=False, kind="mergesort")
    master_unique = master_sorted.drop_duplicates(key_cols, keep="first").copy()

    # Create a lookup keyed by the stable record_id (+ community where available).
    lookup_cols = key_cols + [master_text]
    if master_date:
        lookup_cols.append(master_date)
    if master_year:
        lookup_cols.append(master_year)
    lookup = master_unique[lookup_cols].copy()
    rename_map = {master_text: "original_text"}
    if master_date:
        rename_map[master_date] = "master_date"
    if master_year:
        rename_map[master_year] = "master_year"
    lookup = lookup.rename(columns=rename_map)

    ref = reference.copy()
    # Normalize join keys to strings to avoid mixed dtype mismatches.
    for c in ref_key_cols:
        ref[c] = ref[c].fillna("").astype(str)
    for c in key_cols:
        lookup[c] = lookup[c].fillna("").astype(str)

    merged = ref.merge(lookup, left_on=ref_key_cols, right_on=key_cols, how="left", validate="many_to_one")
    if "original_text" not in merged.columns:
        raise RuntimeError("Original text merge did not produce original_text.")
    missing_original = merged["original_text"].isna() | merged["original_text"].eq("")
    if missing_original.any():
        print(f"WARNING: {int(missing_original.sum())} rows have no matching original text; falling back to {ref_text}.")
        merged.loc[missing_original, "original_text"] = merged.loc[missing_original, ref_text].fillna("").astype(str)

    merged["topic_clean_text"] = merged["original_text"].map(clean_original_text)
    merged["topic_clean_token_count"] = merged["topic_clean_text"].str.split().str.len()
    eligible = merged[merged["topic_clean_token_count"] >= MIN_CLEAN_TOKENS].copy()
    excluded_short = merged[merged["topic_clean_token_count"] < MIN_CLEAN_TOKENS].copy()

    if len(eligible) < K:
        raise ValueError(f"Only {len(eligible)} eligible documents; cannot fit K={K}.")

    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        min_df=5,
        max_df=0.75,
        max_features=15000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-z]{2,}\b",
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(eligible["topic_clean_text"])
    vocab = np.asarray(vectorizer.get_feature_names_out())

    has_features = np.asarray(X.getnnz(axis=1)).ravel() > 0
    model_df = eligible.loc[has_features].copy().reset_index(drop=True)
    X_model = X[has_features]
    unmodelled = eligible.loc[~has_features].copy()

    if X_model.shape[0] < K or X_model.shape[1] < K:
        raise ValueError(f"Insufficient usable matrix for K={K}: {X_model.shape}")

    model = NMF(
        n_components=K,
        init="nndsvda",
        random_state=RANDOM_STATE,
        max_iter=400,
        tol=1e-4,
    )
    W = model.fit_transform(X_model)
    H = model.components_
    row_sums = W.sum(axis=1, keepdims=True)
    shares = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)
    dominant = shares.argmax(axis=1)
    max_share = shares.max(axis=1)

    meta_cols = []
    for c in ["record_id", "id", "community", "subreddit", "date", "year", "source_file"]:
        if c in model_df.columns:
            meta_cols.append(c)
    # Add source text explicitly for review, and keep original + cleaned versions.
    doc_out = model_df[meta_cols].copy()
    doc_out["original_text"] = model_df["original_text"].values
    doc_out["topic_clean_text"] = model_df["topic_clean_text"].values
    doc_out["topic_clean_token_count"] = model_df["topic_clean_token_count"].values
    for i in range(K):
        doc_out[f"topic_{i+1}_weight"] = W[:, i]
        doc_out[f"topic_{i+1}_share"] = shares[:, i]
    doc_out["dominant_topic"] = dominant + 1
    doc_out["dominant_topic_share"] = max_share
    doc_out.to_csv(OUT / f"phase7e_document_topics_k{K}.csv", index=False)

    term_rows = []
    for ti, weights in enumerate(H):
        for rank, idx in enumerate(np.argsort(weights)[::-1][:TOP_TERMS], start=1):
            term_rows.append({
                "topic": ti + 1,
                "rank": rank,
                "term": vocab[idx],
                "term_weight": float(weights[idx]),
            })
    pd.DataFrame(term_rows).to_csv(OUT / f"phase7e_topic_terms_k{K}.csv", index=False)

    # Choose representatives by topic share, then longer text to avoid trivial
    # one-word documents winning every tie. Include source text and clean text.
    reps = []
    for ti in range(K):
        ranked = doc_out.copy()
        ranked["candidate_topic_share"] = shares[:, ti]
        ranked = ranked.sort_values(
            ["candidate_topic_share", "topic_clean_token_count"],
            ascending=[False, False],
            kind="mergesort",
        ).head(TOP_DOCS_PER_TOPIC)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            reps.append({
                "topic": ti + 1,
                "rank": rank,
                "topic_share": row["candidate_topic_share"],
                "topic_clean_token_count": row["topic_clean_token_count"],
                "record_id": row.get("record_id", ""),
                "community": row.get("community", row.get("subreddit", "")),
                "year": row.get("year", ""),
                "original_text": row["original_text"],
                "topic_clean_text": row["topic_clean_text"],
            })
    pd.DataFrame(reps).to_csv(OUT / f"phase7e_representative_documents_k{K}.csv", index=False)

    community_col = find_column(doc_out, ["community", "subreddit"])
    if community_col:
        dist = doc_out.groupby([community_col, "dominant_topic"]).size().rename("document_count").reset_index()
        totals = dist.groupby(community_col)["document_count"].transform("sum")
        dist["within_community_share"] = dist["document_count"] / totals
        dist.to_csv(OUT / f"phase7e_community_distribution_k{K}.csv", index=False)

    if len(excluded_short):
        excluded_short.to_csv(OUT / "phase7e_excluded_short_documents.csv", index=False)
    if len(unmodelled):
        unmodelled.to_csv(OUT / "phase7e_unmodelled_documents.csv", index=False)

    diag = [
        "PHASE 7E NMF — ORIGINAL-TEXT PREPROCESSING",
        f"Master corpus: {MASTER_PATH}",
        f"Phase 7B reference: {PHASE7B_PATH}",
        f"Master rows: {len(master)}",
        f"Master unique key rows after longest-text selection: {len(master_unique)}",
        f"Reference rows: {len(reference)}",
        f"Rows without matching original text (before fallback): {int(missing_original.sum())}",
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
        "Cautions:",
        "- Original corpus is read-only; all transformations are topic-model-only.",
        "- Exact stable record_id + community keys are used; Reddit comment id alone is not used.",
        "- Topics are lexical mixtures and require manual interpretation.",
        "- Reconstruction error is not directly comparable across different document sets/vocabularies.",
        "- Check the missing-original-text count and exclusions before interpreting results.",
    ]
    (OUT / "phase7e_diagnostics.txt").write_text("\n".join(diag), encoding="utf-8")
    print("\n".join(diag))
    print(f"\nOutputs saved to: {OUT}")

if __name__ == "__main__":
    main()
