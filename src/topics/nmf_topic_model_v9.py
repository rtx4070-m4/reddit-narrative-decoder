"""Phase 7H: targeted Reddit NMF cleaning and sensitivity analysis.

Run from project root:
    python src/topics/nmf_topic_model_v9.py                 # audit only
    python src/topics/nmf_topic_model_v9.py --run-model      # fit K=5,6,8,10

Reads Phase 7E input and writes to a NEW Phase 7H directory; Phase 7G is preserved.
"""
from __future__ import annotations
import argparse, re, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables" / "phase7e_nmf" / "phase7e_document_topics_k8.csv"
OUT = ROOT / "results" / "tables" / "phase7h_nmf"
K_VALUES = [5, 6, 8, 10]
RANDOM_STATE = 42
MAX_ITER = 800

# Remove common conversation fillers observed in Phase 7G terms, but preserve
# negation and content-bearing verbs. This is a targeted stoplist, not a claim
# that these words are universally uninformative.
CUSTOM_STOP = {
    "like", "just", "know", "want", "really", "thing", "things", "guy", "guys",
    "actually", "basically", "literally", "anyway", "anyways", "yeah", "yep", "nah",
    "lol", "lmao", "omg", "edit", "update", "thanks", "thank", "please", "sorry",
    "im", "ive", "id", "ill", "youre", "theyre", "weve",
    "thats", "theres", "heres", "whats", "lets", "aint",
    "wwwyoutubecomwatch", "youtubecomwatch", "youtube", "youtu", "http", "https",
    "www", "com", "org", "net", "amp", "gt", "nbsp", "ve", "re", "nt",
}
STOP_WORDS = sorted((set(ENGLISH_STOP_WORDS) - {"no", "nor", "not"}) | CUSTOM_STOP)

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co\.uk|com\.au|org\.uk)\b", re.I)
AUTOTLDR_MARKERS = [
    r"this is the best tl\s*;?\s*dr i could make",
    r"this is the best tl dr i could make",
    r"i am a bot,? and this action was performed automatically",
    r"i'm a bot,? and this action was performed automatically",
    r"please contact the moderators of this subreddit",
    r"reduced by \d+%",
    r"i am a bot",
    r"i'm a bot",
]
AUTOTLDR_RE = re.compile("|".join(f"(?:{p})" for p in AUTOTLDR_MARKERS), re.I)
# Common template fragments; remove only these phrases, not the whole summary.
TEMPLATE_RE = re.compile(
    r"\b(?:extended summary|version \d+|faq version|keywords|constructive welcome|"
    r"read admin|admin comment|comment replies|please read the rules|"
    r"i am a bot|i'm a bot|this action was performed automatically)\b", re.I
)

DOMAIN_HINTS = (
    "www", "http", "https", "youtubecom", "youtu.be", "redditcom", "redd.it",
    "naturecom", "bbccom", "nytimescom", "theguardiancom", "washingtonpostcom",
    "reuterscom", "cnncom", "foxnewscom", "newsworld", "comnews", "co.uk",
)

def flattened_url_token(token: str) -> bool:
    t = token.lower().strip(".,:;!?()[]{}<>\"' ")
    if not t:
        return False
    if t.startswith(("http", "www", "https")):
        return True
    if any(h in t for h in DOMAIN_HINTS):
        return True
    # Long single tokens with domain-like suffixes are often punctuation-stripped URLs.
    if len(t) >= 36 and re.search(r"(?:com|org|net|edu|gov|co|io|uk|au)$", t):
        return True
    # Long mixed alphanumeric token with multiple digit/letter transitions is suspicious.
    if len(t) >= 55 and sum(ch.isdigit() for ch in t) >= 2:
        return True
    return False

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).lower()
    # URLs before punctuation normalization; source data may already have flattened URLs.
    s = URL_RE.sub(" ", s)
    s = DOMAIN_RE.sub(" ", s)
    # Remove explicit AutoTLDR boilerplate markers and common template phrases, preserving
    # substantive summary sentences surrounding them.
    s = AUTOTLDR_RE.sub(" ", s)
    s = re.sub(r"\b(?:tl\s*;?\s*dr|tldr)\b", " ", s, flags=re.I)
    s = TEMPLATE_RE.sub(" ", s)
    s = re.sub(r"\b(?:faq|pms|drs|keywords|moderators|subreddit)\b", " ", s)
    # Expand common contractions before token filtering, retaining negation as explicit tokens.
    contractions = {
        "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not",
        "didn't": "did not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "wouldn't": "would not", "couldn't": "could not",
        "shouldn't": "should not", "i'm": "i am", "you're": "you are", "we're": "we are",
        "they're": "they are", "i've": "i have", "we've": "we have", "they've": "they have",
        "i'll": "i will", "you'll": "you will", "it's": "it is", "that's": "that is",
        "there's": "there is", "what's": "what is", "let's": "let us",
    }
    for old, new in contractions.items():
        s = re.sub(r"\b" + re.escape(old) + r"\b", new, s)
    # Keep alphabetic tokens; identify and discard flattened URL-like tokens individually.
    raw_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", s)
    tokens = []
    for token in raw_tokens:
        token = token.replace("'", "")
        if flattened_url_token(token):
            continue
        if token.isdigit() or len(token) < 2:
            continue
        tokens.append(token)
    return " ".join(tokens)

def first_existing(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find any expected column among {candidates}. Found: {df.columns.tolist()}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-model", action="store_true", help="Fit NMF after writing the Phase 7H audit")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}\nExpected Phase 7E input file. Do not use Phase 7G outputs as input.")
    df = pd.read_csv(INPUT, low_memory=False)
    text_col = first_existing(df, ["original_text", "text", "body", "selftext"])
    id_col = first_existing(df, ["record_id", "id"], required=False)
    community_col = first_existing(df, ["community", "subreddit"], required=False)
    if id_col is None:
        df["record_id_phase7h"] = np.arange(len(df)).astype(str)
        id_col = "record_id_phase7h"
    if community_col is None:
        df["community_phase7h"] = "unknown"
        community_col = "community_phase7h"
    df["original_text"] = df[text_col].fillna("").astype(str)
    df["topic7h_clean_text"] = df["original_text"].map(clean_text)
    df["topic7h_token_count"] = df["topic7h_clean_text"].map(lambda s: len(s.split()) if s else 0)
    df["topic7h_eligible"] = df["topic7h_token_count"] >= 3
    df["source_autotldr_marker"] = df["original_text"].str.contains(AUTOTLDR_RE, na=False)
    df["clean_autotldr_marker"] = df["topic7h_clean_text"].str.contains(AUTOTLDR_RE, na=False)
    df["clean_contains_flattened_url_hint"] = df["topic7h_clean_text"].map(
        lambda s: any(flattened_url_token(t) for t in s.split())
    )
    df["clean_contains_custom_stop"] = df["topic7h_clean_text"].map(
        lambda s: any(t in CUSTOM_STOP for t in s.split())
    )

    # One row per source record/community key. Keep the row with the longest original text
    # if the input unexpectedly contains repeated keys.
    key = df[id_col].astype(str) + "||" + df[community_col].astype(str)
    df["_key"] = key
    df["_original_len"] = df["original_text"].str.len()
    df = df.sort_values(["_original_len"], ascending=False).drop_duplicates("_key", keep="first").drop(columns=["_original_len"])
    eligible = df[df["topic7h_eligible"]].copy()
    unique_texts = eligible["topic7h_clean_text"].drop_duplicates()
    audit = [
        ("input_rows", len(pd.read_csv(INPUT, usecols=[text_col]))),
        ("unique_record_community_keys_after_dedup", len(df)),
        ("eligible_rows_min_3_tokens", int(df["topic7h_eligible"].sum())),
        ("excluded_rows_short_text", int((~df["topic7h_eligible"]).sum())),
        ("unique_cleaned_texts_eligible", int(unique_texts.nunique())),
        ("source_rows_with_autotldr_markers", int(df["source_autotldr_marker"].sum())),
        ("clean_rows_still_with_autotldr_markers", int(df["clean_autotldr_marker"].sum())),
        ("clean_rows_with_flattened_url_hints", int(df["clean_contains_flattened_url_hint"].sum())),
        ("clean_rows_containing_custom_stop_terms", int(df["clean_contains_custom_stop"].sum())),
    ]
    pd.DataFrame(audit, columns=["metric", "value"]).to_csv(OUT / "phase7h_cleaning_audit.csv", index=False)
    examples = df[df["source_autotldr_marker"] | df["clean_contains_flattened_url_hint"]].copy()
    if len(examples) < 100:
        examples = pd.concat([examples, df.sample(min(100-len(examples), len(df)), random_state=RANDOM_STATE)], ignore_index=True).drop_duplicates("_key")
    example_cols = [c for c in [id_col, community_col, "source_autotldr_marker", "clean_autotldr_marker", "topic7h_token_count", "original_text", "topic7h_clean_text"] if c in examples.columns]
    examples[example_cols].head(250).to_csv(OUT / "phase7h_cleaning_examples.csv", index=False)
    # Full cleaning audit preserves metadata and old columns for traceability.
    df.drop(columns=["_key"]).to_csv(OUT / "phase7h_cleaned_documents_audit.csv", index=False)
    print("PHASE 7H CLEANING AUDIT COMPLETE")
    print(pd.DataFrame(audit, columns=["metric", "value"]).to_string(index=False))
    print(f"Audit files saved to: {OUT}")
    if not args.run_model:
        print("NMF NOT FITTED. Inspect phase7h_cleaning_examples.csv first.")
        print("When satisfied, run: python src/topics/nmf_topic_model_v9.py --run-model")
        return

    fit_df = eligible[eligible["topic7h_clean_text"].str.strip().ne("")].copy()
    fit_texts = fit_df["topic7h_clean_text"].drop_duplicates().tolist()
    vectorizer = TfidfVectorizer(
        stop_words=STOP_WORDS, min_df=5, max_df=0.80, max_features=15000,
        ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", dtype=np.float64,
    )
    X_unique = vectorizer.fit_transform(fit_texts)
    terms = np.asarray(vectorizer.get_feature_names_out())
    print(f"Unique-text TF-IDF matrix: {X_unique.shape[0]} x {X_unique.shape[1]}")
    # Transform all eligible rows using exactly the fitted vectorizer.
    X_all = vectorizer.transform(fit_df["topic7h_clean_text"].tolist())
    has_features = np.asarray(X_all.getnnz(axis=1)).ravel() > 0
    fit_df = fit_df.loc[has_features].copy().reset_index(drop=True)
    X_all = X_all[has_features]
    unique_docs = fit_df["topic7h_clean_text"].drop_duplicates().tolist()
    X_unique = vectorizer.transform(unique_docs)
    comparisons, all_terms, all_docs, all_communities, representatives = [], [], [], [], []
    for k in K_VALUES:
        print(f"Fitting K={k} with max_iter={MAX_ITER} ...")
        model = NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE,
                    max_iter=MAX_ITER, tol=1e-4, solver="cd", beta_loss="frobenius")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            W_unique = model.fit_transform(X_unique)
        convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        W_all = model.transform(X_all)
        sums = W_all.sum(axis=1, keepdims=True)
        shares = np.divide(W_all, sums, out=np.zeros_like(W_all), where=sums != 0)
        dominant = shares.argmax(axis=1) + 1
        dominant_share = shares.max(axis=1)
        counts = pd.Series(dominant).value_counts().sort_index().to_dict()
        comparisons.append({"k": k, "reconstruction_error": float(model.reconstruction_err_),
                            "iterations": int(model.n_iter_), "convergence_warning": convergence_warning,
                            "documents_assigned": len(fit_df), "unique_texts_fit": len(unique_docs),
                            "median_dominant_topic_share": float(np.median(dominant_share)),
                            "mean_dominant_topic_share": float(np.mean(dominant_share)),
                            "dominant_topic_counts": str(counts)})
        for topic_idx, component in enumerate(model.components_, start=1):
            top_idx = np.argsort(component)[::-1][:20]
            for rank, ti in enumerate(top_idx, start=1):
                all_terms.append({"k": k, "topic": topic_idx, "rank": rank,
                                  "term": terms[ti], "weight": float(component[ti])})
        doc_out = fit_df[[c for c in [id_col, community_col, "subreddit", "date", "year", "source_file", "original_text", "topic7h_clean_text", "topic7h_token_count"] if c in fit_df.columns]].copy()
        doc_out.insert(0, "k", k)
        for j in range(k):
            doc_out[f"topic_{j+1}_weight"] = W_all[:, j]
            doc_out[f"topic_{j+1}_share"] = shares[:, j]
        doc_out["dominant_topic"] = dominant
        doc_out["dominant_topic_share"] = dominant_share
        all_docs.append(doc_out)
        comm = fit_df[community_col].fillna("unknown").astype(str).to_numpy()
        for community in sorted(set(comm)):
            ix = comm == community
            for topic in range(1, k+1):
                all_communities.append({"k": k, "community": community, "topic": topic,
                                        "document_count": int(np.sum(dominant[ix] == topic)),
                                        "community_documents": int(ix.sum()),
                                        "within_community_share": float(np.mean(dominant[ix] == topic)) if ix.sum() else 0.0})
        doc_out["_row_ix"] = np.arange(len(doc_out))
        for topic in range(1, k+1):
            cand = doc_out[doc_out["dominant_topic"] == topic].sort_values("dominant_topic_share", ascending=False).head(8)
            for rank, (_, row) in enumerate(cand.iterrows(), start=1):
                representatives.append({"k": k, "topic": topic, "rank": rank,
                    "dominant_topic_share": row["dominant_topic_share"],
                    "community": row.get(community_col, "unknown"),
                    "topic7h_token_count": row.get("topic7h_token_count", np.nan),
                    "original_text": row.get("original_text", ""),
                    "topic7h_clean_text": row.get("topic7h_clean_text", "")})
        print(f"  reconstruction error: {model.reconstruction_err_:.4f}; iterations: {model.n_iter_}; convergence warning: {convergence_warning}")
    pd.DataFrame(comparisons).to_csv(OUT / "phase7h_model_comparison.csv", index=False)
    pd.DataFrame(all_terms).to_csv(OUT / "phase7h_topic_terms.csv", index=False)
    pd.concat(all_docs, ignore_index=True).to_csv(OUT / "phase7h_document_topics.csv", index=False)
    pd.DataFrame(all_communities).to_csv(OUT / "phase7h_topic_community_distribution.csv", index=False)
    pd.DataFrame(representatives).to_csv(OUT / "phase7h_representative_documents.csv", index=False)
    (OUT / "phase7h_run_manifest.txt").write_text(
        f"Input: {INPUT}\nOutput: {OUT}\nK values: {K_VALUES}\nRandom state: {RANDOM_STATE}\n"
        f"max_iter: {MAX_ITER}\nstopword count: {len(STOP_WORDS)}\nfit documents: {len(fit_df)}\n"
        f"unique cleaned texts fitted: {len(unique_docs)}\nfeatures: {X_unique.shape[1]}\n", encoding="utf-8")
    print(f"PHASE 7H NMF COMPLETE. Outputs saved to: {OUT}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
