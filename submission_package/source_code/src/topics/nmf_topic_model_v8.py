"""Phase 7G: audited NMF topic-model sensitivity experiment.

This script reads Phase 7E document rows, creates topic-model-only cleaned text,
and writes all Phase 7G outputs to results/tables/phase7g_nmf/.

Safety: default mode is an audit only. Run NMF explicitly with --run-model after
reviewing phase7g_cleaning_audit.csv and phase7g_cleaning_examples.csv.

Run from the project root:
    python src/topics/nmf_topic_model_v8.py                 # audit only
    python src/topics/nmf_topic_model_v8.py --run-model      # audit + fit NMF

Dependencies: pandas, numpy, scikit-learn
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables" / "phase7e_nmf" / "phase7e_document_topics_k8.csv"
OUT = ROOT / "results" / "tables" / "phase7g_nmf"

RANDOM_STATE = 42
TOPIC_COUNTS = [5, 6, 8, 10]
MIN_DF = 3
MAX_DF = 0.80
MAX_FEATURES = 15000
NGRAM_RANGE = (1, 2)
MAX_ITER = 250
MIN_TOKENS = 3
TOP_TERMS = 15

# Topic-model-only stopwords. Keep negation words (not/no/never) available.
EXTRA_STOPWORDS = {
    "http", "https", "www", "com", "org", "net", "gov", "edu", "co", "uk",
    "reddit", "redd", "subreddit", "amp", "gt", "nbsp", "quot", "apos",
    "autotldr", "tldr", "tl", "dr", "feedback", "messagecompose", "fmfaq",
    "bfhttheory", "original", "source", "reduced", "bot", "watch",
    "youtubecomwatch", "npredditcomrautotldr", "like", "just", "know", "want",
}
PRESERVE_STOPWORDS = {
    "no", "nor", "not", "never", "neither", "cannot", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't",
    "couldn't", "shouldn't", "can't",
}
CUSTOM_STOPWORDS = (set(ENGLISH_STOP_WORDS) | EXTRA_STOPWORDS) - PRESERVE_STOPWORDS

# Standard URLs are removed before tokenization. The source data often lost URL
# punctuation earlier, so flattened-domain checks below are intentionally separate.
URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+|\b[a-z0-9.-]+\.(?:com|org|net|gov|edu|co\.uk|com\.au|co\.in)\S*",
    re.I,
)
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")

# Tokens with these substrings are strongly suggestive of flattened domain names.
# This list is deliberately explicit; it only removes the suspicious token, not
# surrounding article-summary text.
FLATTENED_DOMAIN_MARKERS = (
    "naturecom", "washingtonpostcom", "bbccom", "abcnet", "colombogazettecom",
    "independentcouk", "reuterscom", "rtcom", "bigstoryaporg", "economictimes",
    "ibtimes", "youtubecom", "youtu be", "redditcom", "npreddit", "twittercom",
    "imgurcom", "wikipediaorg", "facebookcom", "googlecom", "timesofindia",
    "steamchartscom", "dwcom", "aporg", "newsworld", "comnews", "orgarticle",
)
DOMAIN_SUFFIXES = ("com", "org", "net", "gov", "edu", "couk", "comau", "coin")

# Remove only known AutoTLDR framing/metadata. Preserve the substantive summary.
AUTOTLDR_PATTERNS = [
    r"\bthis is the best tl\s*dr i could make\b",
    r"\bbest tl\s*dr i could make\b",
    r"\breduced by\s+\d+\s*%?\b",
    r"\bi am a bot\b",
    r"\bi'm a bot\b",
    r"\bautotldr\b",
    r"\bnpredditcomrautotl\w*\b",
]

CONTRACTIONS = [
    (re.compile(r"\bcan not\b"), "cannot"),
    (re.compile(r"\bdo not\b"), "don't"),
    (re.compile(r"\bdoes not\b"), "doesn't"),
    (re.compile(r"\bdid not\b"), "didn't"),
    (re.compile(r"\bwill not\b"), "won't"),
    (re.compile(r"\bwould not\b"), "wouldn't"),
    (re.compile(r"\bcould not\b"), "couldn't"),
    (re.compile(r"\bshould not\b"), "shouldn't"),
    (re.compile(r"\bis not\b"), "isn't"),
    (re.compile(r"\bare not\b"), "aren't"),
    (re.compile(r"\bwas not\b"), "wasn't"),
    (re.compile(r"\bwere not\b"), "weren't"),
    (re.compile(r"\bdo nt\b"), "don't"),
    (re.compile(r"\bdoes nt\b"), "doesn't"),
    (re.compile(r"\bdid nt\b"), "didn't"),
    (re.compile(r"\bca nt\b"), "can't"),
    (re.compile(r"\bwo nt\b"), "won't"),
    (re.compile(r"\bwould nt\b"), "wouldn't"),
    (re.compile(r"\bcould nt\b"), "couldn't"),
    (re.compile(r"\bshould nt\b"), "shouldn't"),
    (re.compile(r"\bis nt\b"), "isn't"),
    (re.compile(r"\bare nt\b"), "aren't"),
    (re.compile(r"\bwas nt\b"), "wasn't"),
    (re.compile(r"\bwere nt\b"), "weren't"),
    (re.compile(r"\bi m\b"), "i'm"),
    (re.compile(r"\byou re\b"), "you're"),
    (re.compile(r"\bthey re\b"), "they're"),
    (re.compile(r"\bwe re\b"), "we're"),
    (re.compile(r"\bit s\b"), "it's"),
    (re.compile(r"\bthat s\b"), "that's"),
    (re.compile(r"\bthere s\b"), "there's"),
    (re.compile(r"\bwhat s\b"), "what's"),
    (re.compile(r"\bwho s\b"), "who's"),
    (re.compile(r"\bi ve\b"), "i've"),
    (re.compile(r"\byou ve\b"), "you've"),
    (re.compile(r"\bwe ve\b"), "we've"),
    (re.compile(r"\bthey ve\b"), "they've"),
    (re.compile(r"\bi ll\b"), "i'll"),
    (re.compile(r"\byou ll\b"), "you'll"),
    (re.compile(r"\bwe ll\b"), "we'll"),
    (re.compile(r"\bthey ll\b"), "they'll"),
]


def looks_like_flattened_url(token: str) -> bool:
    """Flag a single token likely created from a URL/domain after punctuation loss."""
    token = token.lower()
    if len(token) >= 45:
        return True
    if any(marker.replace(" ", "") in token for marker in FLATTENED_DOMAIN_MARKERS):
        return True
    # Domain endings alone are only used for long tokens to reduce false positives.
    if len(token) >= 16 and token.endswith(DOMAIN_SUFFIXES):
        return True
    return False


def clean_for_topics(value: object) -> str:
    """Topic-only cleaning; source/original_text remains unchanged."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).lower()
    text = URL_RE.sub(" ", text)

    for pattern in AUTOTLDR_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.I)
    for pattern, replacement in CONTRACTIONS:
        text = pattern.sub(replacement, text)

    cleaned = []
    for token in TOKEN_RE.findall(text):
        if token.isdigit() or len(token) < 2:
            continue
        if looks_like_flattened_url(token):
            continue
        if token in CUSTOM_STOPWORDS:
            continue
        cleaned.append(token)
    return " ".join(cleaned)


def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT}\nRun from the repository after Phase 7E is available."
        )
    df = pd.read_csv(INPUT, low_memory=False)
    required = {"record_id", "community", "original_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df["original_text"] = df["original_text"].fillna("").astype(str)
    df["topic7g_clean_text"] = df["original_text"].map(clean_for_topics)
    df["topic7g_token_count"] = df["topic7g_clean_text"].str.split().str.len()
    df["topic7g_eligible"] = df["topic7g_token_count"] >= MIN_TOKENS

    marker_pattern = r"best tl\s*dr|reduced by\s+\d+|i am a bot|i'm a bot|autotldr"
    source_marker = df["original_text"].str.contains(marker_pattern, case=False, regex=True, na=False)
    clean_marker = df["topic7g_clean_text"].str.contains(marker_pattern, case=False, regex=True, na=False)
    df["source_autotldr_marker"] = source_marker
    df["clean_autotldr_marker"] = clean_marker

    # Save a compact audit table, including whether cleaned rows still contain markers.
    audit = pd.DataFrame([
        {"metric": "input_rows", "value": len(df)},
        {"metric": "unique_record_community_keys", "value": df[["record_id", "community"]].drop_duplicates().shape[0]},
        {"metric": "eligible_rows_min_3_tokens", "value": int(df["topic7g_eligible"].sum())},
        {"metric": "excluded_rows_short_text", "value": int((~df["topic7g_eligible"]).sum())},
        {"metric": "unique_cleaned_texts_all_rows", "value": int(df["topic7g_clean_text"].nunique())},
        {"metric": "source_rows_with_autotldr_markers", "value": int(source_marker.sum())},
        {"metric": "clean_rows_still_with_autotldr_markers", "value": int(clean_marker.sum())},
        {"metric": "clean_rows_containing_like", "value": int(df["topic7g_clean_text"].str.contains(r"\blike\b", regex=True).sum())},
        {"metric": "clean_rows_containing_just", "value": int(df["topic7g_clean_text"].str.contains(r"\bjust\b", regex=True).sum())},
        {"metric": "clean_rows_containing_know", "value": int(df["topic7g_clean_text"].str.contains(r"\bknow\b", regex=True).sum())},
        {"metric": "clean_rows_containing_want", "value": int(df["topic7g_clean_text"].str.contains(r"\bwant\b", regex=True).sum())},
    ])
    return df, audit


def run_model(df: pd.DataFrame) -> None:
    eligible = df.loc[df["topic7g_eligible"]].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No documents remain after topic cleaning.")

    unique_docs = eligible[["topic7g_clean_text"]].drop_duplicates().reset_index(drop=True)
    fit_texts = unique_docs["topic7g_clean_text"].tolist()

    print("PHASE 7G — AUDITED, DEDUPLICATED-FIT NMF SENSITIVITY EXPERIMENT")
    print(f"Input rows: {len(df)}")
    print(f"Eligible rows: {len(eligible)}")
    print(f"Unique cleaned texts used for fitting: {len(unique_docs)}")
    print(f"Repeated eligible rows not separately weighted during fitting: {len(eligible) - len(unique_docs)}")

    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-z][a-z']+\b",
    )
    X_unique = vectorizer.fit_transform(fit_texts)
    terms = np.asarray(vectorizer.get_feature_names_out())
    if X_unique.shape[1] < max(TOPIC_COUNTS):
        warnings.warn(f"Vocabulary has {X_unique.shape[1]} terms; some K values may be too large.")
    print(f"Unique-text TF-IDF matrix: {X_unique.shape[0]} x {X_unique.shape[1]}")

    X_all = vectorizer.transform(eligible["topic7g_clean_text"].tolist())
    coverage = int(np.asarray(X_all.getnnz(axis=1) > 0).sum())
    print(f"Eligible documents with retained features: {coverage}/{len(eligible)}")

    summaries, all_terms, all_assignments = [], [], []
    for k in TOPIC_COUNTS:
        if k >= X_unique.shape[0]:
            print(f"Skipping K={k}: K must be smaller than unique fitting-document count.")
            continue
        print(f"\nFitting K={k} ...")
        model = NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE,
                    max_iter=MAX_ITER, tol=1e-4)
        model.fit(X_unique)
        H = model.components_
        W_all = model.transform(X_all)
        row_sums = W_all.sum(axis=1, keepdims=True)
        shares = np.divide(W_all, row_sums, out=np.zeros_like(W_all), where=row_sums != 0)
        dominant = shares.argmax(axis=1) + 1
        max_share = shares.max(axis=1)
        no_feature = np.asarray(X_all.getnnz(axis=1) == 0)
        topic_counts = pd.Series(dominant).value_counts().reindex(range(1, k + 1), fill_value=0)

        for topic_idx in range(k):
            top_idx = np.argsort(H[topic_idx])[::-1][:TOP_TERMS]
            for rank, term_idx in enumerate(top_idx, start=1):
                all_terms.append({"k": k, "topic": topic_idx + 1, "rank": rank,
                                  "term": terms[term_idx], "weight": float(H[topic_idx, term_idx])})

        result = eligible.copy()
        result["model_k"] = k
        result["dominant_topic"] = dominant
        result["dominant_topic_share"] = max_share
        result["no_retained_features"] = no_feature
        for topic_idx in range(k):
            result[f"topic_{topic_idx + 1}_weight"] = W_all[:, topic_idx]
            result[f"topic_{topic_idx + 1}_share"] = shares[:, topic_idx]
        all_assignments.append(result)

        term_sets = [set(terms[np.argsort(H[i])[::-1][:10]]) for i in range(k)]
        diversity = len(set().union(*term_sets)) / (10 * k) if term_sets else np.nan
        entropy = -sum((count / len(dominant)) * np.log(count / len(dominant))
                       for count in topic_counts if count > 0)
        normalized_entropy = entropy / np.log(k) if k > 1 else np.nan
        summaries.append({
            "k": k,
            "unique_fit_documents": X_unique.shape[0],
            "eligible_assignment_documents": len(eligible),
            "documents_with_features": int((~no_feature).sum()),
            "vocabulary_size": X_unique.shape[1],
            "reconstruction_error_on_unique_fit": float(model.reconstruction_err_),
            "iterations": int(model.n_iter_),
            "dominant_topic_share_median": float(np.median(max_share)),
            "dominant_topic_share_mean": float(np.mean(max_share)),
            "dominant_share_ge_0_90": float(np.mean(max_share >= 0.90)),
            "topic_size_min": int(topic_counts.min()),
            "topic_size_max": int(topic_counts.max()),
            "normalized_topic_size_entropy": float(normalized_entropy),
            "top_term_diversity_10": float(diversity),
            "topic_top10_terms": " || ".join(
                f"T{i+1}: {' '.join(terms[np.argsort(H[i])[::-1][:10]])}" for i in range(k)
            ),
        })
        print(f"  reconstruction error: {model.reconstruction_err_:.4f}")
        print(f"  iterations: {model.n_iter_}")
        print(f"  dominant-topic counts: {topic_counts.to_dict()}")
        print(f"  median dominant share: {np.median(max_share):.4f}")

    if not summaries:
        raise RuntimeError("No candidate K was fitted.")
    pd.DataFrame(summaries).to_csv(OUT / "phase7g_model_comparison.csv", index=False)
    pd.DataFrame(all_terms).to_csv(OUT / "phase7g_topic_terms.csv", index=False)
    assignments = pd.concat(all_assignments, ignore_index=True)
    assignments.to_csv(OUT / "phase7g_document_topics.csv", index=False)

    dist = assignments.groupby(["model_k", "community", "dominant_topic"]).size().rename("documents").reset_index()
    dist["community_share"] = dist["documents"] / dist.groupby(["model_k", "community"])["documents"].transform("sum")
    dist.to_csv(OUT / "phase7g_topic_community_distribution.csv", index=False)

    reps = []
    for k, group in assignments.groupby("model_k"):
        for topic, topic_group in group.groupby("dominant_topic"):
            topic_group = topic_group.sort_values("dominant_topic_share", ascending=False).drop_duplicates("topic7g_clean_text")
            for rank, (_, row) in enumerate(topic_group.head(10).iterrows(), start=1):
                reps.append({"k": int(k), "topic": int(topic), "rank": rank,
                             "dominant_topic_share": float(row["dominant_topic_share"]),
                             "community": row.get("community", ""),
                             "topic7g_token_count": int(row["topic7g_token_count"]),
                             "original_text": row["original_text"],
                             "topic7g_clean_text": row["topic7g_clean_text"]})
    pd.DataFrame(reps).to_csv(OUT / "phase7g_representative_documents.csv", index=False)

    manifest = [
        "Phase 7G audited NMF sensitivity experiment", f"Input: {INPUT}",
        f"Input rows: {len(df)}", f"Eligible rows: {int(df['topic7g_eligible'].sum())}",
        f"Unique cleaned fitting documents: {df.loc[df['topic7g_eligible'], 'topic7g_clean_text'].nunique()}",
        f"Minimum cleaned tokens: {MIN_TOKENS}", f"Candidate K: {TOPIC_COUNTS}",
        f"min_df: {MIN_DF}", f"max_df: {MAX_DF}", f"max_features: {MAX_FEATURES}",
        f"ngram_range: {NGRAM_RANGE}", f"max_iter: {MAX_ITER}", f"random_state: {RANDOM_STATE}",
        "Deduplication scope: unique cleaned text for fitting only",
        "Negation terms are preserved; all cleaning is topic-model-only.",
        "Original corpus and Phase 7E/7F outputs were not modified.",
    ]
    (OUT / "phase7g_run_manifest.txt").write_text("\n".join(manifest), encoding="utf-8")
    print(f"\nNMF outputs saved to: {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7G NMF experiment (audit-only by default).")
    parser.add_argument("--run-model", action="store_true", help="Fit candidate NMF models after generating audit files.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, audit = load_and_clean()
    audit.to_csv(OUT / "phase7g_cleaning_audit.csv", index=False)

    examples = df.loc[
        df["source_autotldr_marker"] | df["clean_autotldr_marker"],
        ["record_id", "community", "source_autotldr_marker", "clean_autotldr_marker",
         "topic7g_token_count", "original_text", "topic7g_clean_text"],
    ].head(250)
    examples.to_csv(OUT / "phase7g_cleaning_examples.csv", index=False)
    # Save a row-level clean-text file for review; no topic model is fitted by default.
    df.to_csv(OUT / "phase7g_cleaned_documents_audit.csv", index=False)

    print("PHASE 7G CLEANING AUDIT COMPLETE (NMF NOT FITTED)")
    print(audit.to_string(index=False))
    print(f"\nAudit files saved to: {OUT}")
    print("Review phase7g_cleaning_audit.csv and phase7g_cleaning_examples.csv first.")
    if args.run_model:
        run_model(df)
    else:
        print("To fit NMF after reviewing the audit, run: python src/topics/nmf_topic_model_v8.py --run-model")


if __name__ == "__main__":
    main()
