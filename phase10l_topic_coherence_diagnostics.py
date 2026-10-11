
from pathlib import Path
from itertools import combinations
from collections import Counter
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
INPUT = (
    ROOT / "results" / "tables"
    / "phase10k_expanded_topic_validation"
    / "phase10k_expanded_sample.csv"
)
OUT_DIR = ROOT / "results" / "tables" / "phase10l_coherence_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[a-z]{3,}", re.IGNORECASE)

STOPWORDS = set("""
the and for that with this from have were was are you your they them
then than there their what when where which would could should have
has had not but all any some more most into about just like know think
really very one two three get got make made use used using people
thing things time way good great bad well also only still even much
many can will did does doing its our out who how why she her his him
""".split())


def top_terms(texts, limit=10):
    counts = Counter()
    for text in texts:
        tokens = [
            t.lower() for t in TOKEN_RE.findall(str(text))
            if t.lower() not in STOPWORDS
        ]
        counts.update(tokens)
    return ", ".join(t for t, _ in counts.most_common(limit))


def main():
    print("=" * 72)
    print("PHASE 10L: TOPIC-COHERENCE DIAGNOSTICS")
    print("=" * 72)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)

    required = {"model", "community", "topic", "record_id", "review_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    df = df[df["review_text"].ne("")].copy()

    if df["record_id"].duplicated().any():
        # The same record may occur in both models, but not twice within one
        # model-community-topic group.
        if df.duplicated(["model", "record_id"]).any():
            raise ValueError("Duplicate record ID within a model.")

    results = []

    for (model, community, topic), group in df.groupby(
        ["model", "community", "topic"], sort=True
    ):
        texts = group["review_text"].tolist()
        n = len(texts)
        unique_texts = len(set(texts))

        result = {
            "model": model,
            "community": community,
            "topic": topic,
            "sample_size": n,
            "available_assigned_examples": int(
                group["available_assigned_examples"].iloc[0]
            ),
            "unique_texts": unique_texts,
            "duplicate_text_fraction": (
                1 - unique_texts / n if n else np.nan
            ),
            "mean_pairwise_tfidf_similarity": np.nan,
            "median_pairwise_tfidf_similarity": np.nan,
            "distinctive_terms": "",
            "diagnostic_note": "",
        }

        # Very small groups cannot support stable pairwise diagnostics.
        if n < 3:
            result["diagnostic_note"] = (
                "Too few sampled texts for a stable similarity summary."
            )
            result["distinctive_terms"] = top_terms(texts)
            results.append(result)
            continue

        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform(texts)

            if matrix.shape[1] == 0:
                result["diagnostic_note"] = (
                    "No usable terms after vectorization."
                )
                results.append(result)
                continue

            sim = cosine_similarity(matrix)
            upper = sim[np.triu_indices(n, k=1)]

            result["mean_pairwise_tfidf_similarity"] = float(
                np.mean(upper)
            )
            result["median_pairwise_tfidf_similarity"] = float(
                np.median(upper)
            )
            result["distinctive_terms"] = top_terms(texts)

            if np.mean(upper) >= 0.25:
                result["diagnostic_note"] = (
                    "Higher lexical similarity; inspect for boilerplate."
                )
            elif np.mean(upper) >= 0.10:
                result["diagnostic_note"] = (
                    "Moderate lexical similarity; review text examples."
                )
            else:
                result["diagnostic_note"] = (
                    "Low lexical similarity; topic may be heterogeneous."
                )

        except ValueError as exc:
            result["diagnostic_note"] = (
                "Vectorization failed: " + str(exc)[:160]
            )

        results.append(result)

    output = pd.DataFrame(results)

    output_path = OUT_DIR / "phase10l_topic_coherence_diagnostics.csv"
    output.to_csv(output_path, index=False)

    comparison = (
        output.groupby("model", as_index=False)
        .agg(
            topic_groups=("topic", "size"),
            sampled_texts=("sample_size", "sum"),
            mean_similarity=("mean_pairwise_tfidf_similarity", "mean"),
            median_similarity=("median_pairwise_tfidf_similarity", "median"),
            mean_duplicate_fraction=("duplicate_text_fraction", "mean"),
        )
    )

    comparison_path = OUT_DIR / "phase10l_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("Topic groups:", len(output))
    print("Sampled records:", int(output["sample_size"].sum()))
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nSaved:", output_path)
    print("Saved:", comparison_path)

    print("\nInterpretation cautions:")
    print("- Similarity is lexical, not semantic truth.")
    print("- Scores are calculated on samples, not full clusters.")
    print("- Higher similarity can reflect boilerplate or repetition.")
    print("- Do not select a model using similarity alone.")

    print("\n" + "=" * 72)
    print("PHASE 10L COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()
