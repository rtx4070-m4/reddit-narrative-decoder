
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "results" / "tables" / "phase10o_human_validation"
OUTPUT_DIR = ROOT / "results" / "tables" / "phase10o_a_ai_review_draft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLES_FILE = INPUT_DIR / "phase10o_example_rating_sheet.csv"
GROUPS_FILE = INPUT_DIR / "phase10o_group_decision_sheet.csv"

# These are preliminary review aids based on earlier exploratory
# examination of sampled examples. They are NOT validated topic labels.
suggestions = [
    # NMF
    ("NMF", "gaming", "Topic 1", "Gameplay experiences and item use", "mixed", "low", "provisional",
     "Examples may mix gameplay experiences with reposts and reactions."),
    ("NMF", "gaming", "Topic 2", "Mixed gaming references and commentary", "mixed", "low", "provisional",
     "Potentially combines gaming references, characters, and unrelated commentary."),
    ("NMF", "gaming", "Topic 3", "Reposts, GIFs, and shared gaming content", "mixed", "moderate", "provisional",
     "Repost and GIF-related language may form a recurring subtheme."),
    ("NMF", "gaming", "Topic 4", "Game characters and gameplay reactions", "mixed", "low", "provisional",
     "Character and gameplay reactions may overlap with memes and general reactions."),
    ("NMF", "gaming", "Topic 5", "General gaming discussion and reactions", "mixed", "low", "provisional",
     "Broad conversational terms may not identify a specific topic."),
    ("NMF", "science", "Topic 1", "General science and technology discussion", "mixed", "low", "provisional",
     "The candidate theme is broad; inspect examples for a more specific shared subject."),
    ("NMF", "science", "Topic 2", "Scientific paper sharing", "unclear", "low", "do_not_label",
     "Very few assigned examples; insufficient evidence for a stable label."),
    ("NMF", "science", "Topic 3", "Gravity and gravitational phenomena", "clear", "moderate", "provisional",
     "Earlier sample review suggested repeated discussion of gravity and related physics."),
    ("NMF", "science", "Topic 4", "Automation and employment", "mixed", "moderate", "provisional",
     "Automation and work-related discussion may recur, but inspect for mixed content."),
    ("NMF", "science", "Topic 5", "Scientific evidence and interpretation", "mixed", "low", "provisional",
     "Evidence-related language may span several unrelated subjects."),
    ("NMF", "todayilearned", "Topic 1", "Factual information and reactions", "mixed", "low", "provisional",
     "May capture reactions to facts rather than one substantive topic."),
    ("NMF", "todayilearned", "Topic 2", "Technology references and general reactions", "mixed", "low", "provisional",
     "Potentially heterogeneous; inspect the actual examples."),
    ("NMF", "todayilearned", "Topic 3", "Short conversational reactions", "mixed", "low", "provisional",
     "Short posts may cluster by style rather than shared subject matter."),
    ("NMF", "todayilearned", "Topic 4", "Political and social commentary", "mixed", "low", "provisional",
     "The proposed theme may combine distinct social and political subjects."),
    ("NMF", "todayilearned", "Topic 5", "Art, intelligence, and other discussions", "mixed", "low", "provisional",
     "Candidate terms suggest multiple potentially unrelated subjects."),
    ("NMF", "worldnews", "Topic 1", "General reactions to news", "mixed", "low", "provisional",
     "May represent general commentary rather than a specific news theme."),
    ("NMF", "worldnews", "Topic 2", "Russia, Syria, and international affairs", "clear", "moderate", "provisional",
     "Earlier samples suggested a recurring geopolitical theme."),
    ("NMF", "worldnews", "Topic 3", "Syria, Turkey, and regional conflict", "clear", "moderate", "provisional",
     "Earlier samples suggested a conflict-related theme; check for overlap with Topic 2."),
    ("NMF", "worldnews", "Topic 4", "General news discussion and reactions", "mixed", "low", "provisional",
     "The topic may be driven by conversational language."),
    ("NMF", "worldnews", "Topic 5", "Geography, politics, and online commentary", "mixed", "low", "provisional",
     "Candidate theme appears broad and potentially heterogeneous."),

    # Semantic K-Means
    ("Semantic KMeans", "gaming", "Topic 1", "Video-game titles, missions, and gameplay", "mixed", "moderate", "provisional",
     "Gameplay is plausible, but the theme may include varied narrative discussion."),
    ("Semantic KMeans", "gaming", "Topic 2", "Game character and narrative reactions", "unclear", "low", "do_not_label",
     "Small group; insufficient evidence for a stable topic label."),
    ("Semantic KMeans", "gaming", "Topic 3", "Gameplay clips and combat reactions", "clear", "moderate", "provisional",
     "Earlier examples suggested gameplay and combat-related content."),
    ("Semantic KMeans", "gaming", "Topic 4", "Mixed gaming and general discussion", "mixed", "low", "provisional",
     "A coherent theme has not yet been established."),
    ("Semantic KMeans", "gaming", "Topic 5", "Spyro and other game references", "mixed", "low", "provisional",
     "The theme may be game-related but could be driven by a narrow set of references."),
    ("Semantic KMeans", "science", "Topic 1", "Automation, employment, and related discussion", "mixed", "low", "provisional",
     "Potentially mixes employment and other subjects."),
    ("Semantic KMeans", "science", "Topic 2", "Paleontology, climate history, and transport policy", "unclear", "low", "do_not_label",
     "Very small group with several candidate subjects."),
    ("Semantic KMeans", "science", "Topic 3", "Requests for explanations and summaries", "unclear", "low", "do_not_label",
     "Very small group; conversational format may dominate."),
    ("Semantic KMeans", "science", "Topic 4", "Health and scientific discussion", "mixed", "low", "provisional",
     "Earlier samples appeared to combine several health and science subjects."),
    ("Semantic KMeans", "science", "Topic 5", "Health, science, and community information", "mixed", "low", "provisional",
     "A broad candidate theme requiring further review."),
    ("Semantic KMeans", "todayilearned", "Topic 1", "Factual posts and general reactions", "mixed", "low", "provisional",
     "Potentially broad rather than a single substantive topic."),
    ("Semantic KMeans", "todayilearned", "Topic 2", "Historical and cultural discussion", "mixed", "low", "provisional",
     "Earlier samples suggested multiple historical and cultural subjects."),
    ("Semantic KMeans", "todayilearned", "Topic 3", "Art and history references", "mixed", "low", "provisional",
     "Small sample relative to group size; verify consistency."),
    ("Semantic KMeans", "todayilearned", "Topic 4", "Historical facts and reactions", "mixed", "low", "provisional",
     "May combine factual content with general reactions."),
    ("Semantic KMeans", "todayilearned", "Topic 5", "Government, culture, and general discussion", "mixed", "low", "provisional",
     "The candidate label covers several possible subjects."),
    ("Semantic KMeans", "worldnews", "Topic 1", "Corporate criticism and political commentary", "mixed", "low", "provisional",
     "Potentially mixes corporate and political subjects."),
    ("Semantic KMeans", "worldnews", "Topic 2", "Government, accountability, and international affairs", "clear", "moderate", "provisional",
     "A possible political and geopolitical theme; inspect examples."),
    ("Semantic KMeans", "worldnews", "Topic 3", "North Korea and sanctions", "clear", "moderate", "provisional",
     "Earlier samples suggested a narrow geopolitical theme; the group is small."),
    ("Semantic KMeans", "worldnews", "Topic 4", "Russia, conflict, and geopolitics", "clear", "moderate", "provisional",
     "Earlier examples suggested recurring geopolitical content."),
    ("Semantic KMeans", "worldnews", "Topic 5", "Europe, the Middle East, and displacement", "mixed", "low", "provisional",
     "Potentially combines several regions and news subjects."),
]

columns = [
    "model", "community", "topic", "suggested_label",
    "suggested_coherence", "suggested_confidence",
    "suggested_decision", "suggestion_basis"
]
lookup = pd.DataFrame(suggestions, columns=columns)

if not EXAMPLES_FILE.exists() or not GROUPS_FILE.exists():
    raise FileNotFoundError(
        "Expected Phase 10O files were not found. Check that Phase 10O "
        "has been run and the two CSVs are in results/tables/phase10o_human_validation."
    )

examples = pd.read_csv(EXAMPLES_FILE)
groups = pd.read_csv(GROUPS_FILE)

keys = ["model", "community", "topic"]

# Standardize model names so the suggestions match the CSV values.
for df in (examples, groups):
    df["model"] = df["model"].astype(str).str.strip().replace({
        "Semantic_KMeans": "Semantic KMeans",
        "semantic_kmeans": "Semantic KMeans",
        "NMF": "NMF",
        "nmf": "NMF",
    })
for df in (examples, groups):
    for col in keys:
        df[col] = df[col].astype(str).str.strip()

# Retain all original columns and add clearly marked suggestion columns.
example_draft = examples.merge(lookup, on=keys, how="left", validate="many_to_one")
group_draft = groups.merge(lookup, on=keys, how="left", validate="one_to_one")

if example_draft["suggested_label"].isna().any():
    missing = example_draft.loc[
        example_draft["suggested_label"].isna(), keys
    ].drop_duplicates()
    raise ValueError(f"Some groups have no suggestion:\n{missing.to_string(index=False)}")

example_path = OUTPUT_DIR / "phase10o_a_example_review_draft.csv"
group_path = OUTPUT_DIR / "phase10o_a_group_review_draft.csv"

example_draft.to_csv(example_path, index=False, encoding="utf-8-sig")
group_draft.to_csv(group_path, index=False, encoding="utf-8-sig")

print("=" * 72)
print("PHASE 10O-A: AI-ASSISTED REVIEW DRAFT")
print("=" * 72)
print(f"Example rows: {len(example_draft)}")
print(f"Topic groups: {len(group_draft)}")
print(f"Suggestions mapped: {group_draft['suggested_label'].notna().sum()}")
print(f"\nSaved: {example_path}")
print(f"Saved: {group_path}")
print("\nIMPORTANT:")
print("- Suggestions are provisional, not human-validation results.")
print("- Existing model assignments have not been changed.")
print("- Human decision columns have not been filled automatically.")
print("- Inspect the actual post examples before accepting any suggestion.")
print("\nPHASE 10O-A COMPLETED")
