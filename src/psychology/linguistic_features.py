from pathlib import Path
import re
import pandas as pd

INPUT_FILE = Path("data/processed/reddit_nlp_corpus.csv")
OUTPUT_DIR = Path("results/tables")
OUTPUT_FILE = OUTPUT_DIR / "linguistic_features.csv"
COMMUNITY_FILE = OUTPUT_DIR / "linguistic_features_community_summary.csv"
YEAR_FILE = OUTPUT_DIR / "linguistic_features_year_summary.csv"

# Transparent word lists: these measure language use, not a person's mental state.
LEXICONS = {
    "first_person": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"},
    "second_person": {"you", "your", "yours", "yourself", "yourselves"},
    "third_person": {"he", "him", "his", "himself", "she", "her", "hers", "herself",
                     "they", "them", "their", "theirs", "themselves", "it", "its", "itself"},
    "negation": {"no", "not", "never", "neither", "nor", "none", "nothing", "nobody",
                 "nowhere", "cannot", "can't", "dont", "don't", "doesnt", "doesn't",
                 "didnt", "didn't", "isnt", "isn't", "wasnt", "wasn't", "wont", "won't",
                 "wouldnt", "wouldn't", "shouldnt", "shouldn't", "couldnt", "couldn't"},
    "cognitive": {"think", "thinks", "thinking", "thought", "know", "knows", "knowing", "knew",
                  "understand", "understands", "understood", "understanding", "realize", "realizes",
                  "realized", "realizing", "believe", "believes", "believed", "believing",
                  "consider", "considers", "considered", "considering", "remember", "remembers",
                  "remembered", "remembering", "forget", "forgets", "forgot", "forgotten",
                  "reason", "reasoning", "reasoned", "explain", "explains", "explained",
                  "explaining", "learn", "learns", "learned", "learning"},
    "uncertainty": {"maybe", "perhaps", "probably", "possibly", "might", "could", "may",
                    "uncertain", "uncertainly", "unlikely", "likely", "guess", "guessing",
                    "assume", "assuming", "apparently", "seems", "seem", "seemed"},
    "social": {"people", "person", "friend", "friends", "family", "parent", "parents", "mother",
               "father", "brother", "sister", "community", "group", "team", "teams", "man",
               "woman", "men", "women", "child", "children", "human", "humans", "society",
               "social", "everyone", "someone", "anyone", "relationship", "relationships"},
    "achievement": {"win", "wins", "won", "winning", "success", "successful", "achieve", "achieved",
                    "achievement", "goal", "goals", "improve", "improved", "improvement", "best",
                    "better", "progress", "performance", "perform", "performed", "score", "scores",
                    "high", "higher", "complete", "completed", "finish", "finished"},
    "risk_threat": {"risk", "risks", "danger", "dangerous", "threat", "threats", "attack", "attacked",
                    "war", "violence", "violent", "death", "dead", "kill", "killed", "harm", "harmful",
                    "crisis", "disaster", "fear", "afraid", "warning", "warn", "problem", "problems",
                    "loss", "lose", "lost"},
    "intensifier": {"very", "really", "extremely", "absolutely", "totally", "completely", "highly",
                    "incredibly", "seriously", "definitely", "clearly", "literally", "so", "too"},
    "hedge": {"maybe", "perhaps", "probably", "possibly", "somewhat", "rather", "fairly", "likely",
              "unlikely", "apparently", "seem", "seems", "seemed", "roughly", "approximately", "generally"},
    "question_word": {"who", "what", "when", "where", "why", "how", "which", "whose", "whom"},
}

# Punctuation-based measures are omitted because the cleaned source text does not
# reliably preserve punctuation/capitalization.
TOKEN_PATTERN = re.compile(r"\b[a-z]+(?:'[a-z]+)?\b", flags=re.IGNORECASE)

def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())

def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0

def extract_features(text):
    text = "" if pd.isna(text) else str(text)
    tokens = tokenize(text)
    n_words = len(tokens)
    unique_words = len(set(tokens))
    result = {
        "word_count": n_words,
        "char_count": len(text),
        "unique_word_count": unique_words,
        "lexical_diversity": ratio(unique_words, n_words),
    }
    for category, lexicon in LEXICONS.items():
        count = sum(token in lexicon for token in tokens)
        result[f"{category}_count"] = count
        result[f"{category}_ratio"] = ratio(count, n_words)

    pronouns = result["first_person_count"] + result["second_person_count"] + result["third_person_count"]
    result["total_pronoun_count"] = pronouns
    result["first_person_share"] = ratio(result["first_person_count"], pronouns)
    result["second_person_share"] = ratio(result["second_person_count"], pronouns)
    result["third_person_share"] = ratio(result["third_person_count"], pronouns)
    return result

def write_group_summary(df, group_col, feature_cols, out_path):
    if group_col not in df.columns:
        print(f"Skipping {group_col} summary: column not found.")
        return
    summary = df.groupby(group_col, dropna=False)[feature_cols].agg(["count", "mean", "median"])
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    summary.reset_index().to_csv(out_path, index=False)

def main():
    print("=" * 70)
    print("PHASE 6A — LINGUISTIC / PSYCHOLOGICAL-LANGUAGE FEATURES")
    print("=" * 70)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}. Run from the project root.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    if "analysis_text" not in df.columns:
        raise ValueError("'analysis_text' column not found. Available columns: " + ", ".join(df.columns.astype(str)))
    if "community" not in df.columns:
        raise ValueError("'community' column not found in the input corpus.")

    print(f"\nInput observations: {len(df):,}")
    rows = []
    for i, text in enumerate(df["analysis_text"], start=1):
        rows.append(extract_features(text))
        if i % 500 == 0 or i == len(df):
            print(f"Processed: {i:,}/{len(df):,}")
    features = pd.DataFrame(rows)

    meta_candidates = ["record_id", "id", "subreddit", "community", "created_utc", "year", "month", "type"]
    meta_cols = [c for c in meta_candidates if c in df.columns]
    result = pd.concat([df[meta_cols].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    feature_cols = list(features.columns)
    result.to_csv(OUTPUT_FILE, index=False)
    write_group_summary(result, "community", feature_cols, COMMUNITY_FILE)
    write_group_summary(result, "year", feature_cols, YEAR_FILE)

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Observations: {len(result):,}")
    print(f"Features created: {len(feature_cols):,}")
    print(f"Missing feature values: {int(result[feature_cols].isna().sum().sum())}")
    if "record_id" in result.columns:
        print(f"Unique record IDs: {result['record_id'].nunique():,}")
    print("\nCommunity sizes:")
    print(result["community"].value_counts(dropna=False).to_string())
    cols = ["word_count", "lexical_diversity", "first_person_ratio", "second_person_ratio",
            "third_person_ratio", "negation_ratio", "cognitive_ratio", "uncertainty_ratio",
            "social_ratio", "achievement_ratio", "risk_threat_ratio", "intensifier_ratio", "hedge_ratio"]
    print("\nMean feature values by community:")
    print(result.groupby("community")[cols].mean().round(4).to_string())
    print("\nSaved:")
    print(f"  {OUTPUT_FILE}\n  {COMMUNITY_FILE}\n  {YEAR_FILE}")
    print("\nThe input/source corpus was not modified.")
    print("Lexicon matches are descriptive language indicators, not diagnoses.")

if __name__ == "__main__":
    main()
