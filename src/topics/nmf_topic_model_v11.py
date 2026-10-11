"""Phase 7J: longer-comment NMF sensitivity experiment.

Run from project root:
    python src/topics/nmf_topic_model_v11.py              # audit only
    python src/topics/nmf_topic_model_v11.py --run-model   # fit K=5 and K=6

Reads Phase 7I's cleaning audit output. Phase 7G/7H/7I outputs are not overwritten.
The main sensitivity change is a minimum of five cleaned tokens plus stricter URL
artifact filtering. Source-flagged AutoTLDR rows are excluded from fitting but
eligible rows can still be transformed into the fitted topic space.
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
INPUT = ROOT / "results" / "tables" / "phase7i_nmf" / "phase7i_cleaned_documents_audit.csv"
OUT = ROOT / "results" / "tables" / "phase7j_nmf"
K_VALUES = [5, 6]
MIN_TOKENS = 5
RANDOM_STATE = 42
MAX_ITER = 800

# Keep the stopword policy conservative. Negation terms are deliberately retained.
CUSTOM_STOP = {
    "like", "just", "know", "want", "really", "thing", "things",
    "actually", "basically", "literally", "anyway", "anyways",
    "yeah", "yep", "nah", "lol", "lmao", "omg", "thanks", "thank",
    "please", "sorry", "im", "ive", "id", "ill", "youre", "theyre",
    "weve", "thats", "theres", "heres", "whats", "lets", "aint",
    "http", "https", "www", "com", "org", "net", "amp", "nbsp",
    "gt", "ve", "re", "nt",
}
STOP_WORDS = sorted((set(ENGLISH_STOP_WORDS) - {"no", "nor", "not"}) | CUSTOM_STOP)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co\.uk|com\.au|org\.uk)\b",
    re.I,
)
TEMPLATE_RE = re.compile(
    r"this is the best tl\s*;?\s*dr i could make|"
    r"\btl\s*;?\s*dr\b|\btldr\b|"
    r"\bi am a bot\b|\bi'm a bot\b|"
    r"\breduced by\s+\d+\b|"
    r"\bthis action was performed automatically\b|"
    r"\bplease contact the moderators\b|"
    r"\b(?:faq version|extended summary|comment replies|read admin|"
    r"constructive welcome|keywords)\b",
    re.I,
)
DOMAIN_HINTS = (
    "naturecom", "youtubecom", "youtu", "redditcom", "nytimescom",
    "theguardiancom", "reuterscom", "bbccom", "cnncom", "wikipediaorg",
    "wikipedia", "newscom", "www", "http", "https",
)

def is_url_artifact(token: str) -> bool:
    t = token.lower().replace("'", "")
    if t.startswith(("http", "https", "www")):
        return True
    if any(h in t for h in DOMAIN_HINTS):
        return True
    # Flattened domains are often long strings with domain suffixes or digits.
    if len(t) >= 28 and re.search(r"(?:com|org|net|edu|gov|co|io|uk|au)\d*$", t):
        return True
    if len(t) >= 45:
        return True
    return False

def clean_text(value: object) -> str:
    s = str(value or "").lower()
    s = URL_RE.sub(" ", s)
    s = DOMAIN_RE.sub(" ", s)
    s = TEMPLATE_RE.sub(" ", s)
    # Split to tokens and remove URL-like fragments after punctuation has been lost.
    raw_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", s)
    tokens = []
    for raw in raw_tokens:
        token = raw.replace("'", "")
        if token.isdigit() or len(token) < 2 or is_url_artifact(token):
            continue
        tokens.append(token)
    return " ".join(tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-model", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run Phase 7I first so its cleaned-document audit exists."
        )

    df = pd.read_csv(INPUT, low_memory=False)
    required = {"record_id", "community", "original_text", "source_autotldr_marker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    df["source_autotldr_marker"] = (
        df["source_autotldr_marker"].fillna(False).astype(str).str.lower()
        .isin(["true", "1", "yes"])
    )
    df["topic7j_clean_text"] = df["original_text"].fillna("").astype(str).map(clean_text)
    df["topic7j_token_count"] = df["topic7j_clean_text"].map(lambda s: len(s.split()))
    df["topic7j_eligible"] = df["topic7j_token_count"] >= MIN_TOKENS
    df["topic7j_residual_template_hint"] = df["topic7j_clean_text"].str.contains(
        r"\b(?:autotldr|reduced|original|keywords|moderators|admin)\b", regex=True, na=False
    )
    df["topic7j_url_artifact_hint"] = df["topic7j_clean_text"].map(
        lambda s: any(is_url_artifact(t) for t in s.split())
    )

    audit = [
        ("input_rows", len(df)),
        ("eligible_rows_min_5_tokens", int(df["topic7j_eligible"].sum())),
        ("excluded_rows_under_5_tokens", int((~df["topic7j_eligible"]).sum())),
        ("source_autotldr_rows_excluded_from_fit", int(df["source_autotldr_marker"].sum())),
        ("eligible_non_autotldr_rows", int((df["topic7j_eligible"] & ~df["source_autotldr_marker"]).sum())),
        ("eligible_autotldr_rows_retained_for_transform_only", int((df["topic7j_eligible"] & df["source_autotldr_marker"]).sum())),
        ("rows_with_residual_template_hints", int(df["topic7j_residual_template_hint"].sum())),
        ("rows_with_url_artifact_hints", int(df["topic7j_url_artifact_hint"].sum())),
        ("unique_cleaned_texts_eligible", int(df.loc[df["topic7j_eligible"], "topic7j_clean_text"].nunique())),
    ]
    audit_df = pd.DataFrame(audit, columns=["metric", "value"])
    audit_df.to_csv(OUT / "phase7j_cleaning_audit.csv", index=False)

    example_mask = (
        df["source_autotldr_marker"] | df["topic7j_residual_template_hint"] |
        df["topic7j_url_artifact_hint"] | ~df["topic7j_eligible"]
    )
    examples = df.loc[example_mask].copy()
    if len(examples) < 120:
        extra = df.sample(min(120 - len(examples), len(df)), random_state=RANDOM_STATE)
        examples = pd.concat([examples, extra], ignore_index=True).drop_duplicates("record_id")
    columns = [c for c in [
        "record_id", "community", "source_autotldr_marker", "original_text",
        "topic7j_clean_text", "topic7j_token_count", "topic7j_eligible",
        "topic7j_residual_template_hint", "topic7j_url_artifact_hint"
    ] if c in examples.columns]
    examples[columns].head(400).to_csv(OUT / "phase7j_cleaning_examples.csv", index=False)
    df.to_csv(OUT / "phase7j_cleaned_documents_audit.csv", index=False)

    print("PHASE 7J CLEANING AUDIT COMPLETE (NMF NOT FITTED)" if not args.run_model
          else "PHASE 7J CLEANING AUDIT COMPLETE")
    print(audit_df.to_string(index=False))
    print(f"Audit files saved to: {OUT}")
    if not args.run_model:
        print("Inspect phase7j_cleaning_examples.csv before fitting.")
        print("When ready: python src/topics/nmf_topic_model_v11.py --run-model")
        return

    eligible = df[df["topic7j_eligible"]].copy().reset_index(drop=True)
    fit_rows = eligible[~eligible["source_autotldr_marker"]].copy()
    fit_unique_texts = fit_rows["topic7j_clean_text"].drop_duplicates().tolist()
    if len(fit_unique_texts) < 20:
        raise ValueError("Too few unique eligible non-AutoTLDR texts to fit NMF reliably.")

    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS, min_df=5, max_df=0.80, max_features=15000,
        ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", dtype=np.float64,
    )
    X_fit = vectorizer.fit_transform(fit_unique_texts)
    terms = np.asarray(vectorizer.get_feature_names_out())
    print(f"Unique non-AutoTLDR texts used for fitting: {len(fit_unique_texts)}")
    print(f"Fit TF-IDF matrix: {X_fit.shape[0]} x {X_fit.shape[1]}")

    X_all = vectorizer.transform(eligible["topic7j_clean_text"].tolist())
    has_features = np.asarray(X_all.getnnz(axis=1)).ravel() > 0
    eligible = eligible.loc[has_features].copy().reset_index(drop=True)
    X_all = X_all[has_features]

    comparisons, topic_terms, document_tables, community_tables, representatives = [], [], [], [], []
    for k in K_VALUES:
        print(f"Fitting K={k} with max_iter={MAX_ITER} ...")
        model = NMF(
            n_components=k, init="nndsvda", random_state=RANDOM_STATE,
            max_iter=MAX_ITER, tol=1e-4, solver="cd", beta_loss="frobenius",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_fit)
        convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        W_all = model.transform(X_all)
        sums = W_all.sum(axis=1, keepdims=True)
        shares = np.divide(W_all, sums, out=np.zeros_like(W_all), where=sums != 0)
        dominant = shares.argmax(axis=1) + 1
        dominant_share = shares.max(axis=1)
        counts = pd.Series(dominant).value_counts().sort_index().to_dict()
        comparisons.append({
            "k": k, "reconstruction_error": float(model.reconstruction_err_),
            "iterations": int(model.n_iter_), "convergence_warning": convergence_warning,
            "fit_unique_texts": len(fit_unique_texts),
            "eligible_documents_transformed": len(eligible),
            "median_dominant_topic_share": float(np.median(dominant_share)),
            "mean_dominant_topic_share": float(np.mean(dominant_share)),
            "dominant_topic_counts_all_eligible": str(counts),
        })

        for topic, component in enumerate(model.components_, start=1):
            for rank, term_idx in enumerate(np.argsort(component)[::-1][:20], start=1):
                topic_terms.append({
                    "k": k, "topic": topic, "rank": rank,
                    "term": terms[term_idx], "weight": float(component[term_idx]),
                })

        keep = [c for c in [
            "record_id", "id", "community", "subreddit", "date", "year",
            "source_file", "original_text", "topic7j_clean_text", "topic7j_token_count",
            "source_autotldr_marker"
        ] if c in eligible.columns]
        out = eligible[keep].copy()
        out.insert(0, "k", k)
        for j in range(k):
            out[f"topic_{j+1}_weight"] = W_all[:, j]
            out[f"topic_{j+1}_share"] = shares[:, j]
        out["dominant_topic"] = dominant
        out["dominant_topic_share"] = dominant_share
        document_tables.append(out)

        for community, group in out.groupby("community", dropna=False):
            for topic in range(1, k + 1):
                community_tables.append({
                    "k": k, "community": str(community), "topic": topic,
                    "document_count": int((group["dominant_topic"] == topic).sum()),
                    "community_documents": len(group),
                    "within_community_share": float((group["dominant_topic"] == topic).mean()),
                })

        for topic in range(1, k + 1):
            candidates = out[out["dominant_topic"] == topic].sort_values(
                "dominant_topic_share", ascending=False
            ).head(10)
            for rank, (_, row) in enumerate(candidates.iterrows(), start=1):
                representatives.append({
                    "k": k, "topic": topic, "rank": rank,
                    "record_id": row.get("record_id", ""),
                    "community": row.get("community", ""),
                    "source_autotldr_marker": row.get("source_autotldr_marker", False),
                    "dominant_topic_share": float(row["dominant_topic_share"]),
                    "original_text": row.get("original_text", ""),
                    "topic7j_clean_text": row.get("topic7j_clean_text", ""),
                })

    pd.DataFrame(comparisons).to_csv(OUT / "phase7j_model_comparison.csv", index=False)
    pd.DataFrame(topic_terms).to_csv(OUT / "phase7j_topic_terms.csv", index=False)
    pd.concat(document_tables, ignore_index=True).to_csv(OUT / "phase7j_document_topics.csv", index=False)
    pd.DataFrame(community_tables).to_csv(OUT / "phase7j_topic_community_distribution.csv", index=False)
    pd.DataFrame(representatives).to_csv(OUT / "phase7j_representative_documents.csv", index=False)
    (OUT / "phase7j_run_manifest.txt").write_text(
        "Phase 7J longer-comment sensitivity experiment\n"
        f"Input: {INPUT}\nOutput: {OUT}\nMinimum cleaned tokens: {MIN_TOKENS}\n"
        f"K values: {K_VALUES}\nRandom state: {RANDOM_STATE}\nMax iterations: {MAX_ITER}\n"
        "AutoTLDR source-flagged rows excluded from fit, eligible rows retained for transform.\n",
        encoding="utf-8",
    )
    print(f"PHASE 7J NMF COMPLETE. Outputs saved to: {OUT}")
    print(pd.DataFrame(comparisons).to_string(index=False))

if __name__ == "__main__":
    main()
