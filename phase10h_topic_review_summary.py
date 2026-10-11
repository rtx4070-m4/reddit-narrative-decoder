
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = (
    ROOT / "results" / "tables"
    / "phase10g_topic_interpretability_audit"
    / "phase10g_sampled_topic_examples.csv"
)
OUT_DIR = ROOT / "results" / "tables" / "phase10h_topic_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("PHASE 10H: TOPIC REVIEW SUMMARY")
    print("=" * 70)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)

    required = {
        "model", "community", "topic", "record_id", "review_text",
        "available_assigned_examples", "human_topic_label",
        "interpretability_rating", "review_notes"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Build one review row for each model-community-topic combination.
    rows = []
    for (model, community, topic), group in df.groupby(
        ["model", "community", "topic"], sort=True
    ):
        rows.append({
            "model": model,
            "community": community,
            "topic": topic,
            "available_assigned_examples":
                int(group["available_assigned_examples"].iloc[0]),
            "examples_in_sample": len(group),
            "proposed_topic_label": "",
            "interpretability_rating": "",
            "evidence_for_label": "",
            "limitations_or_mixed_content": "",
            "decision": "",
        })

    review = pd.DataFrame(rows)

    # Keep a copy of all text examples, with review fields ready to fill.
    examples = df.copy()
    examples["human_topic_label"] = (
        examples["human_topic_label"].fillna("")
    )
    examples["interpretability_rating"] = (
        examples["interpretability_rating"].fillna("")
    )
    examples["review_notes"] = examples["review_notes"].fillna("")

    summary_path = OUT_DIR / "phase10h_topic_review_sheet.csv"
    examples_path = OUT_DIR / "phase10h_examples_for_review.csv"

    review.to_csv(summary_path, index=False)
    examples.to_csv(examples_path, index=False)

    print("Review groups:", len(review))
    print("Sampled examples:", len(examples))
    print("\nGroups by model:")
    print(review.groupby("model").size().to_string())

    print("\nSaved:", summary_path)
    print("Saved:", examples_path)
    print("\nNo source data or model assignments were modified.")
    print("=" * 70)
    print("PHASE 10H COMPLETED")


if __name__ == "__main__":
    main()
