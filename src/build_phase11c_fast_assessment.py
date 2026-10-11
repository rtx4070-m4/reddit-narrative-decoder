import pandas as pd
from pathlib import Path

src = Path("results/tables/phase11c_label_comparison/phase11c_human_review_completed.csv")
out = Path("results/tables/phase11c_label_comparison/phase11c_fast_assessment_draft.csv")

df = pd.read_csv(src).fillna("")

assessments = {
    ("PAIR_1", "A"): (
        "The sample refers to a player character and bottle-smashing gameplay, "
        "but also contains meme-like or conversational language. The available "
        "evidence supports a possible gaming reaction theme, not a definitive "
        "comparison with the other group.",
        "uncertain", "uncertain"
    ),
    ("PAIR_1", "B"): (
        "This group is already marked do_not_label. Preserve that decision. "
        "Its suggested label must not be treated as an accepted topic label.",
        "uncertain", "uncertain"
    ),
    ("PAIR_2", "A"): (
        "The examples include a negative reaction to a game, a suggestion about "
        "a mildly infuriating post, and the phrase shit post. These examples are "
        "broad and do not establish a single coherent topic or equivalence with "
        "the worldnews group.",
        "uncertain", "uncertain"
    ),
    ("PAIR_2", "B"): (
        "The label is broad and lexically resembles the gaming group's label, "
        "but the available comparison fields do not establish whether the two "
        "groups share the same underlying theme. More representative examples "
        "would be needed for a confident conclusion.",
        "uncertain", "uncertain"
    ),
}

for i, row in df.iterrows():
    key = (str(row["pair_id"]), str(row["side"]))
    if key not in assessments:
        continue
    notes, same_theme, recommendation = assessments[key]
    if not str(row.get("human_comparison_notes", "")).strip():
        df.at[i, "human_comparison_notes"] = notes
    if not str(row.get("same_underlying_theme", "")).strip():
        df.at[i, "same_underlying_theme"] = same_theme
    if not str(row.get("label_change_recommendation", "")).strip():
        df.at[i, "label_change_recommendation"] = recommendation

df.to_csv(out, index=False)
print("Draft saved:", out.resolve())
print("Records:", len(df))
print("Existing source file unchanged:", src.exists())
print("Assessment values are provisional suggestions, not completed human validation.")
print(df[["pair_id", "side", "community", "topic",
          "log_decision", "same_underlying_theme",
          "label_change_recommendation"]].to_string(index=False))
