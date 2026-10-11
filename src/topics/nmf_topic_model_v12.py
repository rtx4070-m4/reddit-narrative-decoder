"""Phase 7K: content-focused NMF sensitivity experiment.

Run from project root:
    python src/topics/nmf_topic_model_v12.py
    python src/topics/nmf_topic_model_v12.py --run-model

Uses Phase 7J's cleaned-document audit and preserves all earlier phase outputs.
Controlled comparison:
  A) unigram TF-IDF
  B) unigram + bigram TF-IDF
Both use the same eligible non-AutoTLDR documents, token threshold, stopword policy,
K values, random seed, and NMF settings. This is a sensitivity experiment, not an
automatic final-model selector.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables" / "phase7j_nmf" / "phase7j_cleaned_documents_audit.csv"
OUT = ROOT / "results" / "tables" / "phase7k_nmf"
K_VALUES = [5, 6]
RANDOM_STATE = 42
MAX_ITER = 800
MIN_TOKENS = 5

# Explicit custom terms based on prior topic outputs. Negation is retained.
CUSTOM_STOP = {
    "like", "just", "know", "want", "really", "thing", "things", "actually",
    "basically", "literally", "anyway", "anyways", "yeah", "yep", "nah",
    "lol", "lmao", "omg", "thanks", "thank", "please", "sorry", "im", "ive",
    "id", "ill", "youre", "theyre", "weve", "thats", "theres", "heres",
    "whats", "lets", "aint", "guy", "guys", "stuff", "someone", "anyone",
    "something", "anything", "thing", "things", "http", "https", "www",
    "com", "org", "net", "amp", "nbsp", "gt", "ve", "re", "nt",
    "autotldr", "tldr", "original", "reduced", "keywords", "admin",
    "moderators", "version", "summary", "comment", "comments", "edit",
}
STOP_WORDS = sorted((set(ENGLISH_STOP_WORDS) - {"no", "nor", "not"}) | CUSTOM_STOP)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-model", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run Phase 7J first; Phase 7K reads its cleaned-document audit."
        )

    df = pd.read_csv(INPUT, low_memory=False)
    required = {"record_id", "community", "topic7j_clean_text", "topic7j_token_count",
                "topic7j_eligible", "source_autotldr_marker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df["source_autotldr_marker"] = (
        df["source_autotldr_marker"].fillna(False).astype(str).str.lower()
        .isin(["true", "1", "yes"])
    )
    df["topic7k_clean_text"] = df["topic7j_clean_text"].fillna("").astype(str).str.strip()
    df["topic7k_token_count"] = df["topic7k_clean_text"].map(lambda s: len(s.split()))
    df["topic7k_eligible"] = df["topic7k_token_count"] >= MIN_TOKENS
    df["topic7k_fit_eligible"] = df["topic7k_eligible"] & ~df["source_autotldr_marker"]
    df["topic7k_custom_stopword_occurrences"] = df["topic7k_clean_text"].map(
        lambda s: sum(1 for token in s.split() if token in CUSTOM_STOP)
    )
    df["topic7k_residual_template_hint"] = df["topic7k_clean_text"].str.contains(
        r"\b(?:autotldr|tldr|reduced|original|keywords|moderators|admin)\b",
        regex=True, na=False
    )

    audit = [
        ("input_rows", len(df)),
        ("eligible_rows_min_5_tokens", int(df["topic7k_eligible"].sum())),
        ("excluded_rows_under_5_tokens", int((~df["topic7k_eligible"]).sum())),
        ("source_autotldr_rows_excluded_from_fit", int(df["source_autotldr_marker"].sum())),
        ("eligible_non_autotldr_rows", int(df["topic7k_fit_eligible"].sum())),
        ("eligible_autotldr_rows_retained_for_transform_only",
         int((df["topic7k_eligible"] & df["source_autotldr_marker"]).sum())),
        ("rows_with_residual_template_hints", int(df["topic7k_residual_template_hint"].sum())),
        ("unique_cleaned_texts_eligible", int(df.loc[df["topic7k_eligible"], "topic7k_clean_text"].nunique())),
        ("unique_non_autotldr_texts_for_fit",
         int(df.loc[df["topic7k_fit_eligible"], "topic7k_clean_text"].nunique())),
    ]
    audit_df = pd.DataFrame(audit, columns=["metric", "value"])
    audit_df.to_csv(OUT / "phase7k_cleaning_audit.csv", index=False)

    example_mask = (
        df["source_autotldr_marker"] | df["topic7k_residual_template_hint"] |
        ~df["topic7k_eligible"] | (df["topic7k_custom_stopword_occurrences"] > 0)
    )
    examples = df.loc[example_mask].copy()
    if len(examples) < 150:
        extra = df.sample(min(150 - len(examples), len(df)), random_state=RANDOM_STATE)
        examples = pd.concat([examples, extra], ignore_index=True).drop_duplicates("record_id")
    cols = [c for c in [
        "record_id", "community", "source_autotldr_marker", "original_text",
        "topic7j_clean_text", "topic7k_clean_text", "topic7k_token_count",
        "topic7k_eligible", "topic7k_custom_stopword_occurrences",
        "topic7k_residual_template_hint"
    ] if c in examples.columns]
    examples[cols].head(500).to_csv(OUT / "phase7k_cleaning_examples.csv", index=False)
    df.to_csv(OUT / "phase7k_cleaned_documents_audit.csv", index=False)

    print("PHASE 7K CLEANING AUDIT COMPLETE" + ("" if args.run_model else " (NMF NOT FITTED)"))
    print(audit_df.to_string(index=False))
    print(f"Audit files saved to: {OUT}")
    if not args.run_model:
        print("Inspect phase7k_cleaning_examples.csv before fitting.")
        print("When ready: python src/topics/nmf_topic_model_v12.py --run-model")
        return

    eligible = df[df["topic7k_eligible"]].copy().reset_index(drop=True)
    fit_rows = eligible[~eligible["source_autotldr_marker"]].copy()
    fit_texts = fit_rows["topic7k_clean_text"].drop_duplicates().tolist()
    if len(fit_texts) < 20:
        raise ValueError("Too few unique eligible non-AutoTLDR texts for NMF.")

    # Keep the same document rows and custom stopword policy for both vectorizer variants.
    variants = {
        "unigram": (1, 1),
        "unigram_bigram": (1, 2),
    }
    comparisons, term_rows, doc_rows, community_rows, rep_rows = [], [], [], [], []

    for variant_name, ngram_range in variants.items():
        vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS, min_df=5, max_df=0.80, max_features=15000,
            ngram_range=ngram_range, sublinear_tf=True,
            strip_accents="unicode", dtype=np.float64,
        )
        X_fit = vectorizer.fit_transform(fit_texts)
        terms = np.asarray(vectorizer.get_feature_names_out())
        X_all = vectorizer.transform(eligible["topic7k_clean_text"].tolist())
        has_features = np.asarray(X_all.getnnz(axis=1)).ravel() > 0
        variant_docs = eligible.loc[has_features].copy().reset_index(drop=True)
        X_all = X_all[has_features]
        print(f"\nVariant={variant_name}; fit matrix={X_fit.shape[0]} x {X_fit.shape[1]}; "
              f"transformed rows={len(variant_docs)}")

        for k in K_VALUES:
            print(f"Fitting {variant_name}, K={k}, max_iter={MAX_ITER} ...")
            model = NMF(
                n_components=k, init="nndsvda", random_state=RANDOM_STATE,
                max_iter=MAX_ITER, tol=1e-4, solver="cd", beta_loss="frobenius",
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_fit)
            conv_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            W = model.transform(X_all)
            sums = W.sum(axis=1, keepdims=True)
            shares = np.divide(W, sums, out=np.zeros_like(W), where=sums != 0)
            dominant = shares.argmax(axis=1) + 1
            dominant_share = shares.max(axis=1)
            counts = pd.Series(dominant).value_counts().sort_index().to_dict()

            comparisons.append({
                "variant": variant_name, "ngram_range": str(ngram_range), "k": k,
                "reconstruction_error": float(model.reconstruction_err_),
                "iterations": int(model.n_iter_), "convergence_warning": conv_warning,
                "fit_unique_texts": len(fit_texts), "fit_vocabulary_size": len(terms),
                "eligible_documents_transformed": len(variant_docs),
                "median_dominant_topic_share": float(np.median(dominant_share)),
                "mean_dominant_topic_share": float(np.mean(dominant_share)),
                "dominant_topic_counts": str(counts),
            })

            for topic, component in enumerate(model.components_, start=1):
                for rank, term_idx in enumerate(np.argsort(component)[::-1][:20], start=1):
                    term_rows.append({
                        "variant": variant_name, "k": k, "topic": topic, "rank": rank,
                        "term": terms[term_idx], "weight": float(component[term_idx]),
                    })

            keep = [c for c in [
                "record_id", "id", "community", "subreddit", "date", "year",
                "source_file", "original_text", "topic7k_clean_text",
                "topic7k_token_count", "source_autotldr_marker"
            ] if c in variant_docs.columns]
            out = variant_docs[keep].copy()
            out.insert(0, "variant", variant_name)
            out.insert(1, "k", k)
            for j in range(k):
                out[f"topic_{j+1}_weight"] = W[:, j]
                out[f"topic_{j+1}_share"] = shares[:, j]
            out["dominant_topic"] = dominant
            out["dominant_topic_share"] = dominant_share
            doc_rows.append(out)

            for community, group in out.groupby("community", dropna=False):
                for topic in range(1, k + 1):
                    community_rows.append({
                        "variant": variant_name, "k": k, "community": str(community),
                        "topic": topic,
                        "document_count": int((group["dominant_topic"] == topic).sum()),
                        "community_documents": len(group),
                        "within_community_share": float((group["dominant_topic"] == topic).mean()),
                    })

            for topic in range(1, k + 1):
                candidates = out[out["dominant_topic"] == topic].sort_values(
                    "dominant_topic_share", ascending=False
                ).head(10)
                for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
                    rep_rows.append({
                        "variant": variant_name, "k": k, "topic": topic, "rank": rank,
                        "record_id": row.get("record_id", ""),
                        "community": row.get("community", ""),
                        "source_autotldr_marker": row.get("source_autotldr_marker", False),
                        "dominant_topic_share": float(row["dominant_topic_share"]),
                        "original_text": row.get("original_text", ""),
                        "topic7k_clean_text": row.get("topic7k_clean_text", ""),
                    })

    pd.DataFrame(comparisons).to_csv(OUT / "phase7k_model_comparison.csv", index=False)
    pd.DataFrame(term_rows).to_csv(OUT / "phase7k_topic_terms.csv", index=False)
    pd.concat(doc_rows, ignore_index=True).to_csv(OUT / "phase7k_document_topics.csv", index=False)
    pd.DataFrame(community_rows).to_csv(OUT / "phase7k_topic_community_distribution.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OUT / "phase7k_representative_documents.csv", index=False)
    (OUT / "phase7k_run_manifest.txt").write_text(
        "Phase 7K content-focused NMF sensitivity experiment\n"
        f"Input: {INPUT}\nOutput: {OUT}\nK values: {K_VALUES}\n"
        f"Random state: {RANDOM_STATE}\nMax iterations: {MAX_ITER}\n"
        "Variants: unigram TF-IDF vs unigram+bigram TF-IDF.\n"
        "Same eligible corpus, stopword policy, and NMF settings across variants.\n",
        encoding="utf-8",
    )
    print(f"\nPHASE 7K NMF COMPLETE. Outputs saved to: {OUT}")
    print(pd.DataFrame(comparisons).to_string(index=False))

if __name__ == "__main__":
    main()
