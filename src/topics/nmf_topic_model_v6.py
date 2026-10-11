"""
Phase 7F: Controlled NMF topic-model sensitivity experiment.

Purpose
-------
Compare K=5, 6, 8, and 10 using unique normalized cleaned texts for fitting,
remove identifiable AutoTLDR boilerplate from topic features, then infer
topic mixtures for all eligible Phase 7E documents.

This script does NOT alter the master corpus or Phase 7E outputs.
All transformations are topic-model-only.

Run from the project root:
    python src/topics/nmf_topic_model_v6.py

Dependencies:
    pandas, numpy, scikit-learn
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables" / "phase7e_nmf" / "phase7e_document_topics_k8.csv"
OUT = ROOT / "results" / "tables" / "phase7f_nmf"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TOPIC_COUNTS = [5, 6, 8, 10]
MIN_DF = 3
MAX_DF = 0.80
MAX_FEATURES = 15000
NGRAM_RANGE = (1, 2)
MAX_ITER = 250
MIN_TOKENS = 3

# Keep this conservative. Do not remove negation words globally.
EXTRA_STOPWORDS = {
    "http", "https", "www", "com", "amp", "gt", "nbsp",
    "reddit", "redd", "subreddit",
}

# AutoTLDR boilerplate phrases are removed from topic features only.
AUTOTLDR_PATTERNS = [
    r"\bthis is the best tl\s*dr i could make\b",
    r"\bbest tl\s*dr i could make\b",
    r"\breduced by\s+\d+\b",
    r"\bi am a bot\b",
    r"\bi'm a bot\b",
    r"\bautotldr\b",
]

URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b[a-z0-9.-]+\.(?:com|org|net|gov|edu|co\.uk)\S*", re.I)
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")

# Contractions that remain recognizable in the already-normalized source.
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


def clean_for_topics(value: object) -> str:
    """Conservative topic-only cleaning; original_text remains untouched."""
    text = str(value or "").lower()
    text = URL_RE.sub(" ", text)
    for pattern in AUTOTLDR_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.I)
    for pattern, replacement in CONTRACTIONS:
        text = pattern.sub(replacement, text)
    # Remove obvious URL residue and non-linguistic markup remnants.
    text = re.sub(r"\b(?:original|source)\s+(?:http|https)\b", " ", text)
    tokens = TOKEN_RE.findall(text)
    tokens = [
        tok for tok in tokens
        if tok not in EXTRA_STOPWORDS
        and not tok.isdigit()
        and len(tok) > 1
    ]
    return " ".join(tokens)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT}\n"
            "Run this script from the project repository after Phase 7E."
        )

    df = pd.read_csv(INPUT, low_memory=False)
    required = {"record_id", "community", "original_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df["original_text"] = df["original_text"].fillna("").astype(str)
    df["topic7f_clean_text"] = df["original_text"].map(clean_for_topics)
    df["topic7f_token_count"] = df["topic7f_clean_text"].str.split().str.len()

    eligible = df.loc[df["topic7f_token_count"] >= MIN_TOKENS].copy()
    if eligible.empty:
        raise ValueError("No documents remain after topic cleaning.")

    # Fit once per unique normalized cleaned text to avoid repetition-weighting.
    # Each unique text is represented by its first occurrence for fitting only.
    unique_docs = (
        eligible[["topic7f_clean_text"]]
        .drop_duplicates(subset=["topic7f_clean_text"], keep="first")
        .reset_index(drop=True)
    )
    fit_texts = unique_docs["topic7f_clean_text"].tolist()

    print("PHASE 7F — DEDUPLICATED-FIT NMF SENSITIVITY EXPERIMENT")
    print(f"Input rows: {len(df)}")
    print(f"Eligible documents (>= {MIN_TOKENS} cleaned tokens): {len(eligible)}")
    print(f"Excluded for short cleaned text: {len(df) - len(eligible)}")
    print(f"Unique cleaned texts used for fitting: {len(unique_docs)}")
    print(f"Repeated eligible rows not separately weighted during fitting: {len(eligible) - len(unique_docs)}")
    bot_pattern = r"best tl\s*dr|reduced by\s+\d+|i am a bot|autotldr"
    bot_marker_rows = int(
        df["original_text"].str.contains(bot_pattern, case=False, regex=True).sum()
    )
    print(f"AutoTLDR-marker rows in original text: {bot_marker_rows}")

    vectorizer = TfidfVectorizer(
        lowercase=False,  # text already lowercased by cleaning
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
        warnings.warn(
            f"Vocabulary has {X_unique.shape[1]} terms; some requested K values "
            "may be too large for this matrix."
        )
    print(f"Unique-text TF-IDF matrix: {X_unique.shape[0]} x {X_unique.shape[1]}")

    # Vectorize every eligible document with the exact same fitted vocabulary.
    X_all = vectorizer.transform(eligible["topic7f_clean_text"].tolist())
    coverage = int(np.asarray(X_all.getnnz(axis=1) > 0).sum())
    print(f"Eligible documents with retained features: {coverage}/{len(eligible)}")

    summaries = []
    all_terms = []
    all_assignments = []

    for k in TOPIC_COUNTS:
        if k >= X_unique.shape[0]:
            print(f"Skipping K={k}: K must be smaller than unique fitting-document count.")
            continue
        print(f"\nFitting K={k} ...")
        model = NMF(
            n_components=k,
            init="nndsvda",
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
            tol=1e-4,
        )
        W_unique = model.fit_transform(X_unique)
        H = model.components_
        error = float(model.reconstruction_err_)

        # NMF transform for all eligible documents using this fitted model.
        W_all = model.transform(X_all)
        row_sums = W_all.sum(axis=1, keepdims=True)
        shares = np.divide(W_all, row_sums, out=np.zeros_like(W_all), where=row_sums != 0)
        dominant = shares.argmax(axis=1) + 1
        max_share = shares.max(axis=1)
        no_feature = np.asarray(X_all.getnnz(axis=1) == 0)

        # Topic-term table.
        for topic_idx in range(k):
            top_idx = np.argsort(H[topic_idx])[::-1][:15]
            for rank, term_idx in enumerate(top_idx, start=1):
                all_terms.append({
                    "k": k, "topic": topic_idx + 1, "rank": rank,
                    "term": terms[term_idx], "weight": float(H[topic_idx, term_idx]),
                })

        # Per-document topic weights/shares.
        result = eligible.copy().reset_index(drop=True)
        result["model_k"] = k
        result["dominant_topic"] = dominant
        result["dominant_topic_share"] = max_share
        result["no_retained_features"] = no_feature
        for topic_idx in range(k):
            result[f"topic_{topic_idx + 1}_weight"] = W_all[:, topic_idx]
            result[f"topic_{topic_idx + 1}_share"] = shares[:, topic_idx]
        all_assignments.append(result)

        topic_counts = pd.Series(dominant).value_counts().reindex(range(1, k + 1), fill_value=0)
        top_terms_by_topic = [
            " ".join(terms[np.argsort(H[i])[::-1][:10]]) for i in range(k)
        ]
        # Mean pairwise cosine similarity among top-term TF-IDF vectors is not
        # calculated here; this summary intentionally avoids calling it semantic coherence.
        term_sets = [set(terms[np.argsort(H[i])[::-1][:10]]) for i in range(k)]
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        diversity = (
            len(set().union(*term_sets)) / (10 * k) if k and term_sets else np.nan
        )
        entropy = -sum(
            (count / len(dominant)) * np.log(count / len(dominant))
            for count in topic_counts if count > 0
        )
        normalized_entropy = entropy / np.log(k) if k > 1 else np.nan
        summaries.append({
            "k": k,
            "unique_fit_documents": X_unique.shape[0],
            "eligible_assignment_documents": len(eligible),
            "documents_with_features": int((~no_feature).sum()),
            "vocabulary_size": X_unique.shape[1],
            "reconstruction_error_on_unique_fit": error,
            "iterations": int(model.n_iter_),
            "dominant_topic_share_median": float(np.median(max_share)),
            "dominant_topic_share_mean": float(np.mean(max_share)),
            "dominant_share_ge_0_90": float(np.mean(max_share >= 0.90)),
            "topic_size_min": int(topic_counts.min()),
            "topic_size_max": int(topic_counts.max()),
            "normalized_topic_size_entropy": float(normalized_entropy),
            "top_term_diversity_10": float(diversity),
            "topic_top10_terms": " || ".join(
                f"T{i+1}: {top_terms_by_topic[i]}" for i in range(k)
            ),
        })
        print(f"  reconstruction error: {error:.4f}")
        print(f"  iterations: {model.n_iter_}")
        print(f"  dominant-topic counts: {topic_counts.to_dict()}")
        print(f"  median dominant share: {np.median(max_share):.4f}")

    if not summaries:
        raise RuntimeError("No candidate K was fitted.")

    pd.DataFrame(summaries).to_csv(OUT / "phase7f_model_comparison.csv", index=False)
    pd.DataFrame(all_terms).to_csv(OUT / "phase7f_topic_terms.csv", index=False)
    if all_assignments:
        assignments = pd.concat(all_assignments, ignore_index=True)
        assignments.to_csv(OUT / "phase7f_document_topics.csv", index=False)

        # A compact topic/community distribution table for each K.
        dist = (
            assignments.groupby(["model_k", "community", "dominant_topic"])
            .size().rename("documents").reset_index()
        )
        dist["community_share"] = dist["documents"] / dist.groupby(
            ["model_k", "community"]
        )["documents"].transform("sum")
        dist.to_csv(OUT / "phase7f_topic_community_distribution.csv", index=False)

        # Representative documents: rank by dominant topic share, with distinct
        # text previews so identical repeated texts do not occupy every slot.
        reps = []
        for k, group in assignments.groupby("model_k"):
            for topic, topic_group in group.groupby("dominant_topic"):
                topic_group = topic_group.sort_values(
                    "dominant_topic_share", ascending=False
                ).drop_duplicates("topic7f_clean_text")
                for rank, (_, row) in enumerate(topic_group.head(10).iterrows(), start=1):
                    reps.append({
                        "k": int(k),
                        "topic": int(topic),
                        "rank": rank,
                        "dominant_topic_share": float(row["dominant_topic_share"]),
                        "community": row.get("community", ""),
                        "topic7f_token_count": int(row["topic7f_token_count"]),
                        "original_text": row["original_text"],
                        "topic7f_clean_text": row["topic7f_clean_text"],
                    })
        pd.DataFrame(reps).to_csv(
            OUT / "phase7f_representative_documents.csv", index=False
        )

    # Record experiment parameters for reproducibility.
    (OUT / "phase7f_run_manifest.txt").write_text(
        "\n".join([
            "Phase 7F NMF sensitivity experiment",
            f"Input: {INPUT}",
            f"Input rows: {len(df)}",
            f"Eligible rows: {len(eligible)}",
            f"Unique cleaned fitting documents: {len(unique_docs)}",
            f"Minimum cleaned tokens: {MIN_TOKENS}",
            f"Candidate K: {TOPIC_COUNTS}",
            f"min_df: {MIN_DF}",
            f"max_df: {MAX_DF}",
            f"max_features: {MAX_FEATURES}",
            f"ngram_range: {NGRAM_RANGE}",
            f"max_iter: {MAX_ITER}",
            f"random_state: {RANDOM_STATE}",
            "Deduplication scope: unique cleaned text for fitting only",
            "Original corpus and Phase 7E outputs were not modified",
        ]),
        encoding="utf-8",
    )

    print(f"\nOutputs saved to: {OUT}")
    print("Review phase7f_model_comparison.csv and phase7f_topic_terms.csv first.")
    print("Reconstruction errors are comparable only within this Phase 7F experiment.")


if __name__ == "__main__":
    main()
