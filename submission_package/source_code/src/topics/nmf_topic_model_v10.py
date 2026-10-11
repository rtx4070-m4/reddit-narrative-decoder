"""Phase 7I: AutoTLDR-source sensitivity experiment for Reddit NMF.

Run from project root:
    python src/topics/nmf_topic_model_v10.py                 # audit only
    python src/topics/nmf_topic_model_v10.py --run-model      # fit sensitivity models

Phase 7H is preserved. This experiment excludes source rows flagged as AutoTLDR
from vocabulary/model fitting, but transforms eligible rows (including those
excluded from fitting) into the learned topic space for comparison.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables" / "phase7h_nmf" / "phase7h_cleaned_documents_audit.csv"
OUT = ROOT / "results" / "tables" / "phase7i_nmf"
K_VALUES = [5, 6, 8, 10]
RANDOM_STATE = 42
MAX_ITER = 800

CUSTOM_STOP = {
    "like", "just", "know", "want", "really", "thing", "things", "guy", "guys",
    "actually", "basically", "literally", "anyway", "anyways", "yeah", "yep", "nah",
    "lol", "lmao", "omg", "edit", "update", "thanks", "thank", "please", "sorry",
    "im", "ive", "id", "ill", "youre", "theyre", "weve", "thats", "theres",
    "heres", "whats", "lets", "aint", "wwwyoutubecomwatch", "youtubecomwatch",
    "youtube", "youtu", "http", "https", "www", "com", "org", "net", "amp",
    "gt", "nbsp", "ve", "re", "nt",
}
STOP_WORDS = sorted((set(ENGLISH_STOP_WORDS) - {"no", "nor", "not"}) | CUSTOM_STOP)

# These patterns are used only to clean recognizable template fragments.
# The experiment's primary sensitivity intervention is excluding source-flagged
# AutoTLDR rows from fitting, not deleting substantive summary text.
AUTOTLDR_MARKERS = re.compile(
    r"this is the best tl\s*;?\s*dr i could make|"
    r"\btl\s*;?\s*dr\b|\btldr\b|"
    r"\bi am a bot\b|\bi'm a bot\b|"
    r"\breduced by\s+\d+\b|"
    r"\bthis action was performed automatically\b",
    re.I,
)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co\.uk|com\.au|org\.uk)\b",
    re.I,
)
FLATTENED_HINTS = (
    "www", "http", "https", "youtubecom", "redditcom", "naturecom",
    "bbccom", "nytimescom", "theguardiancom", "reuterscom", "cnncom",
)

def suspicious_flattened_url(token: str) -> bool:
    t = token.lower()
    if t.startswith(("http", "https", "www")):
        return True
    if any(hint in t for hint in FLATTENED_HINTS):
        return True
    if len(t) >= 38 and re.search(r"(?:com|org|net|edu|gov|co|io|uk|au)$", t):
        return True
    if len(t) >= 55 and sum(ch.isdigit() for ch in t) >= 2:
        return True
    return False

def clean_template_fragments(text: str) -> str:
    s = str(text).lower()
    s = URL_RE.sub(" ", s)
    s = DOMAIN_RE.sub(" ", s)
    # Remove source-template phrases while preserving article-summary prose.
    s = re.sub(r"this is the best tl\s*;?\s*dr i could make", " ", s, flags=re.I)
    s = re.sub(r"\b(?:original|reduced by|i am a bot|i'm a bot|"
               r"this action was performed automatically|please contact the moderators)\b",
               " ", s, flags=re.I)
    s = re.sub(r"\b(?:faq version|extended summary|comment replies|"
               r"read admin|constructive welcome|keywords)\b", " ", s, flags=re.I)
    # Remove only URL-like flattened tokens, not ordinary article words.
    raw = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", s)
    kept = []
    for token in raw:
        token = token.replace("'", "")
        if token.isdigit() or len(token) < 2 or suspicious_flattened_url(token):
            continue
        kept.append(token)
    return " ".join(kept)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-model", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run Phase 7H first; do not use the NMF result file as input."
        )

    df = pd.read_csv(INPUT, low_memory=False)
    required = {"record_id", "community", "original_text", "source_autotldr_marker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Phase 7H audit is missing required columns: {sorted(missing)}")

    df["source_autotldr_marker"] = df["source_autotldr_marker"].fillna(False).astype(bool)
    df["topic7i_clean_text"] = df["original_text"].fillna("").astype(str).map(clean_template_fragments)
    df["topic7i_token_count"] = df["topic7i_clean_text"].map(lambda x: len(x.split()))
    df["topic7i_eligible"] = df["topic7i_token_count"] >= 3
    df["topic7i_source_autotldr_excluded_from_fit"] = df["source_autotldr_marker"]
    df["topic7i_residual_template_hint"] = df["topic7i_clean_text"].str.contains(
        r"\b(?:autotldr|reduced by|original|keywords|admin|moderators)\b", regex=True, na=False
    )
    df["topic7i_flattened_url_hint"] = df["topic7i_clean_text"].map(
        lambda s: any(suspicious_flattened_url(t) for t in s.split())
    )

    audit = [
        ("input_rows", len(df)),
        ("eligible_rows_min_3_tokens", int(df["topic7i_eligible"].sum())),
        ("excluded_short_rows", int((~df["topic7i_eligible"]).sum())),
        ("source_autotldr_rows_excluded_from_fit", int(df["source_autotldr_marker"].sum())),
        ("eligible_non_autotldr_rows", int((df["topic7i_eligible"] & ~df["source_autotldr_marker"]).sum())),
        ("eligible_autotldr_rows_retained_for_transform_only", int((df["topic7i_eligible"] & df["source_autotldr_marker"]).sum())),
        ("rows_with_residual_template_hints", int(df["topic7i_residual_template_hint"].sum())),
        ("rows_with_flattened_url_hints", int(df["topic7i_flattened_url_hint"].sum())),
        ("unique_cleaned_texts_eligible", int(df.loc[df["topic7i_eligible"], "topic7i_clean_text"].nunique())),
    ]
    audit_df = pd.DataFrame(audit, columns=["metric", "value"])
    audit_df.to_csv(OUT / "phase7i_cleaning_audit.csv", index=False)

    sample_cols = [
        c for c in ["record_id", "community", "source_autotldr_marker",
                    "topic7i_token_count", "original_text", "topic7i_clean_text",
                    "topic7i_residual_template_hint", "topic7i_flattened_url_hint"]
        if c in df.columns
    ]
    examples = df[df["source_autotldr_marker"] | df["topic7i_residual_template_hint"] |
                  df["topic7i_flattened_url_hint"]].copy()
    if len(examples) < 100:
        extra = df.sample(min(100 - len(examples), len(df)), random_state=RANDOM_STATE)
        examples = pd.concat([examples, extra], ignore_index=True).drop_duplicates("record_id")
    examples[sample_cols].head(300).to_csv(OUT / "phase7i_cleaning_examples.csv", index=False)
    df.to_csv(OUT / "phase7i_cleaned_documents_audit.csv", index=False)

    print("PHASE 7I CLEANING AUDIT COMPLETE")
    print(audit_df.to_string(index=False))
    print(f"Audit files saved to: {OUT}")
    if not args.run_model:
        print("NMF NOT FITTED. Inspect phase7i_cleaning_examples.csv first.")
        print("When satisfied, run: python src/topics/nmf_topic_model_v10.py --run-model")
        return

    eligible = df[df["topic7i_eligible"]].copy().reset_index(drop=True)
    fit_rows = eligible[~eligible["source_autotldr_marker"]].copy()
    if fit_rows.empty:
        raise ValueError("No eligible non-AutoTLDR rows available for model fitting.")

    # Match Phase 7H vectorizer/model settings for a controlled comparison.
    fit_unique = fit_rows["topic7i_clean_text"].drop_duplicates().tolist()
    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS, min_df=5, max_df=0.80, max_features=15000,
        ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", dtype=np.float64,
    )
    X_fit = vectorizer.fit_transform(fit_unique)
    terms = np.asarray(vectorizer.get_feature_names_out())
    print(f"Unique non-AutoTLDR texts used for fitting: {len(fit_unique)}")
    print(f"Fit TF-IDF matrix: {X_fit.shape[0]} x {X_fit.shape[1]}")

    # Transform every eligible row, including source-flagged rows, for comparability.
    X_all = vectorizer.transform(eligible["topic7i_clean_text"].tolist())
    has_features = np.asarray(X_all.getnnz(axis=1)).ravel() > 0
    eligible = eligible.loc[has_features].copy().reset_index(drop=True)
    X_all = X_all[has_features]

    comparisons, terms_out, docs_out, community_out, reps_out = [], [], [], [], []
    for k in K_VALUES:
        print(f"Fitting K={k} with max_iter={MAX_ITER} ...")
        model = NMF(
            n_components=k, init="nndsvda", random_state=RANDOM_STATE,
            max_iter=MAX_ITER, tol=1e-4, solver="cd", beta_loss="frobenius",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            W_fit = model.fit_transform(X_fit)
        warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        W_all = model.transform(X_all)
        row_sums = W_all.sum(axis=1, keepdims=True)
        shares = np.divide(W_all, row_sums, out=np.zeros_like(W_all), where=row_sums != 0)
        dominant = shares.argmax(axis=1) + 1
        dom_share = shares.max(axis=1)
        counts = pd.Series(dominant).value_counts().sort_index().to_dict()
        comparisons.append({
            "k": k, "reconstruction_error": float(model.reconstruction_err_),
            "iterations": int(model.n_iter_), "convergence_warning": warning,
            "fit_unique_texts": len(fit_unique), "eligible_documents_transformed": len(eligible),
            "median_dominant_topic_share": float(np.median(dom_share)),
            "mean_dominant_topic_share": float(np.mean(dom_share)),
            "dominant_topic_counts_all_eligible": str(counts),
        })

        for topic_idx, component in enumerate(model.components_, start=1):
            for rank, idx in enumerate(np.argsort(component)[::-1][:20], start=1):
                terms_out.append({"k": k, "topic": topic_idx, "rank": rank,
                                  "term": terms[idx], "weight": float(component[idx])})

        keep = [c for c in ["record_id", "id", "community", "subreddit", "date", "year",
                            "source_file", "original_text", "topic7i_clean_text",
                            "topic7i_token_count", "source_autotldr_marker"] if c in eligible.columns]
        out = eligible[keep].copy()
        out.insert(0, "k", k)
        for j in range(k):
            out[f"topic_{j+1}_weight"] = W_all[:, j]
            out[f"topic_{j+1}_share"] = shares[:, j]
        out["dominant_topic"] = dominant
        out["dominant_topic_share"] = dom_share
        docs_out.append(out)

        for community, group in out.groupby("community", dropna=False):
            for topic in range(1, k + 1):
                community_out.append({
                    "k": k, "community": str(community), "topic": topic,
                    "document_count": int((group["dominant_topic"] == topic).sum()),
                    "community_documents": len(group),
                    "within_community_share": float((group["dominant_topic"] == topic).mean()),
                })

        for topic in range(1, k + 1):
            candidates = out[out["dominant_topic"] == topic].sort_values(
                "dominant_topic_share", ascending=False
            ).head(8)
            for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
                reps_out.append({
                    "k": k, "topic": topic, "rank": rank,
                    "record_id": row.get("record_id", ""),
                    "community": row.get("community", ""),
                    "source_autotldr_marker": row.get("source_autotldr_marker", False),
                    "dominant_topic_share": row["dominant_topic_share"],
                    "original_text": row.get("original_text", ""),
                    "topic7i_clean_text": row.get("topic7i_clean_text", ""),
                })

    pd.DataFrame(comparisons).to_csv(OUT / "phase7i_model_comparison.csv", index=False)
    pd.DataFrame(terms_out).to_csv(OUT / "phase7i_topic_terms.csv", index=False)
    pd.concat(docs_out, ignore_index=True).to_csv(OUT / "phase7i_document_topics.csv", index=False)
    pd.DataFrame(community_out).to_csv(OUT / "phase7i_topic_community_distribution.csv", index=False)
    pd.DataFrame(reps_out).to_csv(OUT / "phase7i_representative_documents.csv", index=False)
    (OUT / "phase7i_run_manifest.txt").write_text(
        "Phase 7I AutoTLDR source sensitivity experiment\n"
        f"Input: {INPUT}\nOutput: {OUT}\nK values: {K_VALUES}\n"
        f"Random state: {RANDOM_STATE}\nMax iterations: {MAX_ITER}\n"
        "Source-flagged AutoTLDR rows excluded from fitting but retained for transform/assignment.\n"
        "TF-IDF/model settings matched Phase 7H where applicable.\n",
        encoding="utf-8",
    )
    print(f"PHASE 7I NMF COMPLETE. Outputs saved to: {OUT}")
    print(pd.DataFrame(comparisons).to_string(index=False))

if __name__ == "__main__":
    main()
