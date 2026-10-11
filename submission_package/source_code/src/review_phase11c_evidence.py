from pathlib import Path
import pandas as pd

SOURCE = Path(
    "results/tables/phase11c_label_comparison/"
    "phase11c_label_evidence_comparison.csv"
)

OUTPUT_DIR = Path("results/tables/phase11c_label_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "phase11c_human_review_completed.csv"

if not SOURCE.exists():
    raise FileNotFoundError(f"Review file not found: {SOURCE}")

df = pd.read_csv(SOURCE).fillna("")

required = [
    "pair_id", "side", "model", "community", "topic",
    "working_topic_label", "log_decision",
    "representative_sample_texts", "candidate_terms_from_sample",
    "supporting_evidence", "limitations_or_mixed_content",
    "evidence_suggestion_basis",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for col in [
    "human_comparison_notes",
    "same_underlying_theme",
    "label_change_recommendation",
]:
    if col not in df.columns:
        df[col] = ""

def show_field(label, value):
    value = str(value).strip()
    print(f"\n{label}:")
    print(value if value else "[No information recorded in this field]")

def ask_choice(prompt, options, default=None):
    choices = "/".join(options)
    while True:
        suffix = f" [{default}]" if default else ""
        answer = input(f"{prompt} ({choices}){suffix}: ").strip().lower()
        if not answer and default:
            return default
        if answer in options:
            return answer
        print("Please enter one of:", ", ".join(options))

print("=" * 78)
print("PHASE 11C — INTERACTIVE EVIDENCE REVIEW")
print("=" * 78)
print(f"Records to review: {len(df)}")
print("Enter q at a review prompt to save progress and exit.")
print("Existing source labels and decisions will not be changed.")

for idx in range(len(df)):
    row = df.iloc[idx]

    if (
        str(row.get("human_comparison_notes", "")).strip()
        and str(row.get("same_underlying_theme", "")).strip()
        and str(row.get("label_change_recommendation", "")).strip()
    ):
        continue

    print("\n" + "=" * 78)
    print(
        f"RECORD {idx + 1} OF {len(df)} | "
        f"{row['pair_id']} | SIDE {row['side']}"
    )
    print("=" * 78)

    for col, label in [
        ("model", "Model"),
        ("community", "Community"),
        ("topic", "Topic"),
        ("working_topic_label", "Current working label"),
        ("log_decision", "Recorded decision"),
        ("label_status", "Label status"),
        ("candidate_terms_from_sample", "Candidate terms"),
        ("representative_sample_texts", "Representative sample texts"),
        ("supporting_evidence", "Supporting evidence"),
        ("limitations_or_mixed_content", "Limitations or mixed content"),
        ("evidence_suggestion_basis", "Evidence review basis"),
    ]:
        if col in df.columns:
            show_field(label, row[col])

    if str(row["log_decision"]).strip().lower() == "do_not_label":
        print(
            "\nSAFEGUARD: This group has an existing do_not_label decision. "
            "This review will not overwrite that decision."
        )

    while True:
        notes = input(
            "\nYour evidence-based comparison notes "
            "(or q to save and exit): "
        ).strip()
        if notes.lower() == "q":
            df.to_csv(OUTPUT, index=False)
            print(f"\nProgress saved to: {OUTPUT.resolve()}")
            raise SystemExit(0)
        if notes:
            break
        print("Please enter notes, even if your conclusion is uncertain.")

    same = ask_choice(
        "Do these two topics share the same underlying theme?",
        ["yes", "no", "uncertain"],
    )

    recommendation = ask_choice(
        "Recommended label action",
        ["keep", "revise", "do_not_label", "uncertain"],
    )

    # Preserve the original decision. A do_not_label topic cannot be
    # upgraded by this comparison script.
    if str(row["log_decision"]).strip().lower() == "do_not_label":
        if recommendation in {"keep", "revise"}:
            print(
                "\nSafeguard: the recorded decision is do_not_label. "
                "The recommendation is stored as uncertain instead; "
                "a separate, explicit decision process would be needed "
                "to change the recorded decision."
            )
            recommendation = "uncertain"

    df.at[idx, "human_comparison_notes"] = notes
    df.at[idx, "same_underlying_theme"] = same
    df.at[idx, "label_change_recommendation"] = recommendation

    # Save after every reviewed record to prevent loss of progress.
    df.to_csv(OUTPUT, index=False)
    print("Review saved.")

print("\n" + "=" * 78)
print("PHASE 11C REVIEW FINISHED")
print("=" * 78)
print(f"Records: {len(df)}")
print(f"Output: {OUTPUT.resolve()}")
print("Original comparison file was not modified.")
print("Recorded model decisions were not modified.")
