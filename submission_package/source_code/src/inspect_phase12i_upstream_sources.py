from pathlib import Path

src = Path("src")
terms = [
    "linguistic_features.csv",
    "vader_sentiment_results.csv",
    "transformer_emotion_results.csv",
    "reddit_nlp_corpus.csv",
]

print("=== GENERATION AND INPUT REFERENCES ===")

for path in sorted(src.rglob("*.py")):
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        continue

    matches = [
        i for i, line in enumerate(lines)
        if any(term.lower() in line.lower() for term in terms)
    ]

    if not matches:
        continue

    print(f"\n--- {path} ---")
    shown = set()
    for i in matches:
        for j in range(max(0, i - 5), min(len(lines), i + 7)):
            if j not in shown:
                print(f"{j + 1:4}: {lines[j]}")
                shown.add(j)
        print("...")
