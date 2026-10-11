from pathlib import Path

path = Path("src/topics/nmf_topic_model_v8.py")
lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

print("=== PHASE 7G GENERATOR: INPUTS AND FILTERS ===")

keywords = (
    "INPUT", "OUTPUT", "read_csv", "to_csv",
    "dropna", "drop_duplicates", "filter",
    "eligible", "token_count", "original_text",
    "record_id", "def main", "def load",
    "phase7f", "phase7e", "phase7d", "reddit_nlp_corpus",
    "reddit_master_clean"
)

selected = set()

for i, line in enumerate(lines):
    if any(word.lower() in line.lower() for word in keywords):
        for j in range(max(0, i - 3), min(len(lines), i + 5)):
            selected.add(j)

for i in sorted(selected):
    print(f"{i + 1:4}: {lines[i]}")

print("\n=== END OF EXTRACT ===")
