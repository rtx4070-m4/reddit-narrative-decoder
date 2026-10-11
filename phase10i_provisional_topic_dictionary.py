
from pathlib import Path
from collections import Counter
import re
import pandas as pd

# ============================================================
# PHASE 10I: PROVISIONAL TOPIC DICTIONARY
# Uses sampled text to support transparent human topic labeling.
# Does not retrain models or change existing assignments.
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT = (
    ROOT / "results" / "tables"
    / "phase10g_topic_interpretability_audit"
    / "phase10g_sampled_topic_examples.csv"
)

OUT_DIR = ROOT / "results" / "tables" / "phase10i_topic_dictionary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = set("""
a an the and or but if then else of to in on at for from by with about
as is am are was were be been being it its this that these those
i me my mine we us our ours you your yours he him his she her hers
they them their theirs who whom what which when where why how
do does did doing have has had having can could should would will
may might must not no nor so than too very just also really
there here now still even only much many more most some any all
get got make made one two first like know think say said see want
use used using thing things way time day good great bad well
im ive id ill dont doesnt didnt cant couldnt wouldnt shouldnt
isnt wasnt werent arent havent hasnt hadnt youre theyre thats
theres whats whos lets
""".split())

TOKEN_RE = re.compile(r"[a-z][a-z0-9']{2,}", re.IGNORECASE)


def extract_terms(text):
    tokens = [
        token.lower()
        for token in TOKEN_RE.findall(str(text))
    ]
    return [
        token for token in tokens
        if token not in STOPWORDS and not token.isdigit()
    ]


def make_examples(texts, max_chars=240):
    cleaned = []
    for text in texts:
        text = re.sub(r"\s+", " ", str(text)).strip()
        if text and text not in cleaned:
            cleaned.append(text[:max_chars])
    return " || ".join(cleaned[:3])


def main():
    print("=" * 72)
    print("PHASE 10I: PROVISIONAL TOPIC DICTIONARY")
    print("=" * 72)

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False)

    required = {
        "model", "community", "topic", "record_id", "review_text"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["review_text"] = df["review_text"].fillna("").astype(str)
    df = df[df["review_text"].str.strip().ne("")].copy()

    if df.empty:
        raise ValueError("No non-empty sampled text examples found.")

    rows = []

    for (model, community, topic), group in df.groupby(
        ["model", "community", "topic"], sort=True
    ):
        term_counts = Counter()

        for text in group["review_text"]:
            term_counts.update(extract_terms(text))

        top_terms = [
            term for term, count in term_counts.most_common(12)
        ]

        available = (
            int(group["available_assigned_examples"].iloc[0])
            if "available_assigned_examples" in group.columns
            else len(group)
        )

        rows.append({
            "model": model,
            "community": community,
            "topic": topic,
            "available_assigned_examples": available,
            "sampled_examples": len(group),
            "candidate_terms_from_sample": ", ".join(top_terms),
            "representative_sample_texts": make_examples(
                group["review_text"].tolist()
            ),
            "proposed_topic_label": "",
            "label_confidence": "",
            "supporting_evidence": "",
            "limitations_or_mixed_content": "",
            "final_decision": "",
        })

    result = pd.DataFrame(rows)

    output_file = (
        OUT_DIR / "phase10i_provisional_topic_dictionary.csv"
    )
    result.to_csv(output_file, index=False)

    print("Sample records reviewed:", len(df))
    print("Model-community-topic groups:", len(result))
    print("\nGroups per model:")
    print(result.groupby("model").size().to_string())

    print("\nPreview:")
    print(
        result[
            [
                "model", "community", "topic",
                "sampled_examples", "candidate_terms_from_sample"
            ]
        ].head(12).to_string(index=False)
    )

    print("\nSaved:", output_file)
    print("\nImportant:")
    print("- Candidate terms are descriptive, not validated topic labels.")
    print("- Labels and confidence remain blank for human review.")
    print("- Small samples may not represent the full cluster.")
    print("- No source files or model assignments were modified.")

    print("\n" + "=" * 72)
    print("PHASE 10I COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()
