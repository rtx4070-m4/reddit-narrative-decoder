
from pathlib import Path
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PHASE 10M: SEMANTIC COHERENCE VALIDATION
# Compare sentence-embedding similarity within topic groups.
# Uses local embeddings; no external API calls.
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT = (
    ROOT / "results" / "tables"
    / "phase10k_expanded_topic_validation"
    / "phase10k_expanded_sample.csv"
)

OUT_DIR = ROOT / "results" / "tables" / "phase10m_semantic_coherence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16


def main():
    print("=" * 76)
    print("PHASE 10M: SEMANTIC COHERENCE VALIDATION")
    print("=" * 76)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)

    required = {"model", "community", "topic", "record_id", "review_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    df = df[df["review_text"].ne("")].copy()

    # The same source record may be assigned in both models, but should
    # occur no more than once per model.
    if df.duplicated(["model", "record_id"]).any():
        raise ValueError("Duplicate record IDs found within a model.")

    print("Sampled records:", len(df))
    print("Groups:", df.groupby(["model", "community", "topic"]).ngroups)
    print("Loading local model:", MODEL_NAME)
    print("If the model is not cached, Transformers may need to download it.")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        df["review_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    df = df.reset_index(drop=True)
    group_rows = []
    text_rows = []

    for (model_name, community, topic), indexes in df.groupby(
        ["model", "community", "topic"], sort=True
    ).groups.items():
        positions = list(indexes)
        vectors = embeddings[positions]
        group = df.iloc[positions]
        n = len(group)

        result = {
            "model": model_name,
            "community": community,
            "topic": topic,
            "sample_size": n,
            "available_assigned_examples": int(
                group["available_assigned_examples"].iloc[0]
            ),
            "mean_pairwise_semantic_similarity": np.nan,
            "median_pairwise_semantic_similarity": np.nan,
            "minimum_pairwise_semantic_similarity": np.nan,
            "maximum_pairwise_semantic_similarity": np.nan,
            "mean_similarity_to_group_centroid": np.nan,
            "diagnostic_note": "",
        }

        if n < 3:
            result["diagnostic_note"] = (
                "Too few sampled texts for a stable similarity summary."
            )
        else:
            similarities = cosine_similarity(vectors)
            pair_values = similarities[np.triu_indices(n, k=1)]

            centroid = vectors.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)

            if centroid_norm > 0:
                centroid = centroid / centroid_norm
                centroid_scores = vectors @ centroid
                result["mean_similarity_to_group_centroid"] = float(
                    np.mean(centroid_scores)
                )

            result["mean_pairwise_semantic_similarity"] = float(
                np.mean(pair_values)
            )
            result["median_pairwise_semantic_similarity"] = float(
                np.median(pair_values)
            )
            result["minimum_pairwise_semantic_similarity"] = float(
                np.min(pair_values)
            )
            result["maximum_pairwise_semantic_similarity"] = float(
                np.max(pair_values)
            )

            mean_sim = result["mean_pairwise_semantic_similarity"]

            # These are descriptive flags, not universal thresholds.
            if mean_sim >= 0.60:
                result["diagnostic_note"] = (
                    "Relatively high embedding similarity; inspect examples."
                )
            elif mean_sim >= 0.40:
                result["diagnostic_note"] = (
                    "Intermediate embedding similarity; human review needed."
                )
            else:
                result["diagnostic_note"] = (
                    "Lower embedding similarity; check for mixed content."
                )

        group_rows.append(result)

        for local_idx, (_, row) in enumerate(group.iterrows()):
            text_rows.append({
                "model": model_name,
                "community": community,
                "topic": topic,
                "record_id": row["record_id"],
                "review_text": row["review_text"],
                "similarity_to_group_centroid": (
                    float(centroid_scores[local_idx])
                    if n >= 3 and centroid_norm > 0
                    else np.nan
                ),
            })

    diagnostics = pd.DataFrame(group_rows)
    text_diagnostics = pd.DataFrame(text_rows)

    comparison = (
        diagnostics.groupby("model", as_index=False)
        .agg(
            topic_groups=("topic", "size"),
            sampled_records=("sample_size", "sum"),
            mean_pairwise_semantic_similarity=(
                "mean_pairwise_semantic_similarity", "mean"
            ),
            median_group_similarity=(
                "mean_pairwise_semantic_similarity", "median"
            ),
            mean_centroid_similarity=(
                "mean_similarity_to_group_centroid", "mean"
            ),
        )
    )

    diagnostics_path = OUT_DIR / "phase10m_topic_semantic_similarity.csv"
    comparison_path = OUT_DIR / "phase10m_model_comparison.csv"
    text_path = OUT_DIR / "phase10m_record_centroid_similarity.csv"

    diagnostics.to_csv(diagnostics_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    text_diagnostics.to_csv(text_path, index=False)

    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    print("\nTopic groups:", len(diagnostics))
    print("Records:", len(text_diagnostics))
    print("\nSaved:", diagnostics_path)
    print("Saved:", comparison_path)
    print("Saved:", text_path)

    print("\nInterpretation cautions:")
    print("- Similarity scores are descriptive, not ground-truth coherence.")
    print("- Scores depend on the embedding model and the text domain.")
    print("- A high score may reflect boilerplate or common phrasing.")
    print("- Do not infer individual psychological traits from text.")
    print("- Do not select a model using these scores alone.")

    print("\n" + "=" * 76)
    print("PHASE 10M COMPLETED")
    print("=" * 76)


if __name__ == "__main__":
    main()
