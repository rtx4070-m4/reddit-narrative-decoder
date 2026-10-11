
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "results" / "tables"

INPUT = (
    TABLES / "phase10k_expanded_topic_validation"
    / "phase10k_expanded_sample.csv"
)

OUT_DIR = TABLES / "phase10o_human_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 72)
    print("PHASE 10O: HUMAN VALIDATION SHEET")
    print("=" * 72)

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT, low_memory=False)

    required = {
        "model", "community", "topic", "record_id", "review_text"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["review_text"] = df["review_text"].fillna("").astype(str)

    if df.duplicated(["model", "record_id"]).any():
        raise ValueError("Duplicate record IDs within a model.")

    # Preserve all source columns and add standardized human ratings.
    df["human_topic_label"] = ""
    df["coherence_rating"] = ""
    df["label_confidence"] = ""
    df["evidence_notes"] = ""
    df["contradictory_content"] = ""

    df["coherence_rating"] = df["coherence_rating"].astype("object")
    df["label_confidence"] = df["label_confidence"].astype("object")

    sample_path = OUT_DIR / "phase10o_example_rating_sheet.csv"
    df.to_csv(sample_path, index=False)

    # One row per group for the final decision.
    groups = (
        df.groupby(["model", "community", "topic"], sort=True)
        .agg(
            available_records=("available_assigned_examples", "first"),
            sampled_records=("record_id", "count"),
        )
        .reset_index()
    )

    groups["proposed_label"] = ""
    groups["overall_coherence"] = ""
    groups["label_confidence"] = ""
    groups["final_decision"] = ""
    groups["justification"] = ""

    group_path = OUT_DIR / "phase10o_group_decision_sheet.csv"
    groups.to_csv(group_path, index=False)

    print("Sampled examples:", len(df))
    print("Topic groups:", len(groups))
    print("Saved:", sample_path)
    print("Saved:", group_path)

    print("\nUse these ratings:")
    print("coherence_rating: clear / mixed / unclear")
    print("label_confidence: high / moderate / low")
    print("final_decision: retain / provisional / do_not_label")
    print("\nDo not infer psychological traits about authors.")
    print("Existing model assignments remain unchanged.")

    print("\nPHASE 10O COMPLETED")


if __name__ == "__main__":
    main()
