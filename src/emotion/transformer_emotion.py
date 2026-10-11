from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reddit_nlp_corpus.csv"
)

TABLES_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "j-hartmann/"
    "emotion-english-distilroberta-base"
)

BATCH_SIZE = 8
MAX_LENGTH = 256

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TRANSFORMER EMOTION ANALYSIS")
print("=" * 70)

print(
    f"\nModel: {MODEL_NAME}"
)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )


# ============================================================
# LOAD CORPUS
# ============================================================

print("\nLoading corpus...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["created_utc"]
)

print(
    f"Observations: {len(df):,}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "record_id",
    "community",
    "analysis_text",
    "created_utc",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )

if df["analysis_text"].isna().any():

    raise ValueError(
        "analysis_text contains missing values."
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading emotion model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()


# ============================================================
# LABEL MAP
# ============================================================

id2label = model.config.id2label

label_map = {
    int(k): str(v).lower()
    for k, v in id2label.items()
}

print("\nEmotion labels:")

for index in sorted(label_map):

    print(
        f"  {index}: "
        f"{label_map[index]}"
    )


# ============================================================
# TEXTS
# ============================================================

texts = (
    df["analysis_text"]
    .fillna("")
    .astype(str)
    .tolist()
)


# ============================================================
# STORAGE
# ============================================================

all_probabilities = []
predicted_labels = []
predicted_scores = []


# ============================================================
# BATCH INFERENCE
# ============================================================

print("\nRunning transformer inference...")

num_batches = (
    len(texts)
    + BATCH_SIZE
    - 1
) // BATCH_SIZE


with torch.no_grad():

    for batch_number, start in enumerate(
        range(
            0,
            len(texts),
            BATCH_SIZE
        ),
        start=1,
    ):

        batch_texts = texts[
            start:
            start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(
            **encoded
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        probabilities = (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

        predictions = (
            np.argmax(
                probabilities,
                axis=1
            )
        )

        scores = (
            np.max(
                probabilities,
                axis=1
            )
        )

        all_probabilities.extend(
            probabilities
        )

        predicted_labels.extend(
            [
                label_map[int(i)]
                for i in predictions
            ]
        )

        predicted_scores.extend(
            scores
            .tolist()
        )

        if (
            batch_number == 1
            or batch_number % 25 == 0
            or batch_number == num_batches
        ):

            processed = min(
                start + BATCH_SIZE,
                len(texts)
            )

            print(
                f"Batch "
                f"{batch_number:,}/"
                f"{num_batches:,} | "
                f"Processed "
                f"{processed:,}/"
                f"{len(texts):,}"
            )


# ============================================================
# PROBABILITY DATAFRAME
# ============================================================

probabilities_array = np.asarray(
    all_probabilities
)

emotion_columns = [
    label_map[i]
    for i in range(
        len(label_map)
    )
]

probability_df = pd.DataFrame(
    probabilities_array,
    columns=[
        f"emotion_prob_{label}"
        for label in emotion_columns
    ]
)


# ============================================================
# ADD PREDICTIONS
# ============================================================

result = pd.concat(
    [
        df.reset_index(drop=True),
        probability_df,
    ],
    axis=1
)

result["emotion_label"] = (
    predicted_labels
)

result["emotion_confidence"] = (
    predicted_scores
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("EMOTION OUTPUT VALIDATION")
print("=" * 70)

print(
    f"\nRows: {len(result):,}"
)

print(
    f"Probability columns: "
    f"{len(emotion_columns)}"
)

print(
    "\nMissing prediction labels:"
)

print(
    result["emotion_label"]
    .isna()
    .sum()
)

print(
    "\nMissing confidence values:"
)

print(
    result["emotion_confidence"]
    .isna()
    .sum()
)

probability_sum = (
    result[
        [
            f"emotion_prob_{label}"
            for label in emotion_columns
        ]
    ]
    .sum(axis=1)
)

print(
    "\nMaximum probability-sum deviation:"
)

print(
    np.abs(
        probability_sum - 1
    ).max()
)


# ============================================================
# OVERALL EMOTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("OVERALL EMOTION DISTRIBUTION")
print("=" * 70)

overall = (
    result["emotion_label"]
    .value_counts()
    .rename_axis("emotion")
    .reset_index(
        name="observations"
    )
)

overall["percentage"] = (
    overall["observations"]
    / len(result)
    * 100
)

print(
    overall.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

overall.to_csv(
    TABLES_DIR
    / "transformer_emotion_overall.csv",
    index=False
)


# ============================================================
# COMMUNITY EMOTION COUNTS
# ============================================================

community_counts = (
    pd.crosstab(
        result["community"],
        result["emotion_label"]
    )
    .reset_index()
)

print(
    "\nCommunity × emotion counts:"
)

print(
    community_counts.to_string(
        index=False
    )
)

community_counts.to_csv(
    TABLES_DIR
    / "transformer_emotion_community_counts.csv",
    index=False
)


# ============================================================
# COMMUNITY EMOTION PERCENTAGES
# ============================================================

community_percentages = (
    pd.crosstab(
        result["community"],
        result["emotion_label"],
        normalize="index"
    )
    * 100
)

community_percentages = (
    community_percentages
    .reset_index()
)

print(
    "\nCommunity × emotion percentages:"
)

print(
    community_percentages.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

community_percentages.to_csv(
    TABLES_DIR
    / "transformer_emotion_community_percentages.csv",
    index=False
)


# ============================================================
# COMMUNITY PROBABILITY MEANS
# ============================================================

probability_columns = [
    f"emotion_prob_{label}"
    for label in emotion_columns
]

community_probability_means = (
    result.groupby(
        "community",
        observed=True
    )[probability_columns]
    .mean()
    .reset_index()
)

print(
    "\nMean emotion probabilities by community:"
)

print(
    community_probability_means.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

community_probability_means.to_csv(
    TABLES_DIR
    / "transformer_emotion_community_probabilities.csv",
    index=False
)


# ============================================================
# YEAR
# ============================================================

result["year"] = (
    result["created_utc"]
    .dt.year
)

year_counts = (
    pd.crosstab(
        result["year"],
        result["emotion_label"]
    )
    .reset_index()
)

year_counts.to_csv(
    TABLES_DIR
    / "transformer_emotion_year_counts.csv",
    index=False
)


# ============================================================
# COMMUNITY × YEAR
# ============================================================

community_year_counts = (
    pd.crosstab(
        [
            result["community"],
            result["year"]
        ],
        result["emotion_label"]
    )
    .reset_index()
)

community_year_counts.to_csv(
    TABLES_DIR
    / "transformer_emotion_community_year_counts.csv",
    index=False
)


# ============================================================
# HIGH-CONFIDENCE PREDICTIONS
# ============================================================

high_confidence = result[
    result["emotion_confidence"] >= 0.80
].copy()

print(
    f"\nPredictions with "
    f"confidence >= 0.80: "
    f"{len(high_confidence):,}"
)

print(
    f"Percentage: "
    f"{len(high_confidence) / len(result) * 100:.2f}%"
)


# ============================================================
# LOW-CONFIDENCE PREDICTIONS
# ============================================================

low_confidence = result[
    result["emotion_confidence"] < 0.50
].copy()

print(
    f"\nPredictions with "
    f"confidence < 0.50: "
    f"{len(low_confidence):,}"
)

print(
    f"Percentage: "
    f"{len(low_confidence) / len(result) * 100:.2f}%"
)


# ============================================================
# CONFIDENCE SUMMARY
# ============================================================

confidence_summary = pd.DataFrame({
    "metric": [
        "mean",
        "median",
        "minimum",
        "maximum",
        "q1",
        "q3",
    ],
    "value": [
        result["emotion_confidence"].mean(),
        result["emotion_confidence"].median(),
        result["emotion_confidence"].min(),
        result["emotion_confidence"].max(),
        result["emotion_confidence"].quantile(0.25),
        result["emotion_confidence"].quantile(0.75),
    ]
})

print(
    "\nConfidence summary:"
)

print(
    confidence_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

confidence_summary.to_csv(
    TABLES_DIR
    / "transformer_emotion_confidence.csv",
    index=False
)


# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

output_columns = [
    "record_id",
    "id",
    "community",
    "subreddit",
    "created_utc",
    "year",
    "text",
    "analysis_text",

    "emotion_label",
    "emotion_confidence",
]

output_columns.extend(
    probability_columns
)

output_columns.extend([
    "is_very_short",
    "is_article_like",
    "is_repeated_text",
])

emotion_results = result[
    output_columns
]

output_file = (
    TABLES_DIR
    / "transformer_emotion_results.csv"
)

emotion_results.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMER EMOTION ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nObservations analyzed: "
    f"{len(result):,}"
)

print(
    f"Emotions: "
    f"{', '.join(emotion_columns)}"
)

print(
    "\nComplete results:"
)

print(
    output_file
)

print(
    "\nMaster/NLP corpus was NOT modified."
)

print("=" * 70)