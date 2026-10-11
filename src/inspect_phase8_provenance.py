from pathlib import Path

src = Path("src")
terms = [
    "phase8_merged_observations.csv",
    "phase8_merged_observations",
    "phase7j_cleaned_documents_audit.csv",
    "phase7j_cleaned_documents",
    "phase8_cross_community",
]

print("=== SOURCE CODE REFERENCES ===")

for path in sorted(src.rglob("*.py")):
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        continue

    matches = [
        i for i, line in enumerate(lines)
        if any(term.lower() in line.lower() for term in terms)
    ]

    if matches:
        print(f"\n--- {path} ---")
        shown = set()
        for i in matches:
            start = max(0, i - 3)
            end = min(len(lines), i + 5)
            for j in range(start, end):
                if j not in shown:
                    print(f"{j + 1:4}: {lines[j]}")
                    shown.add(j)
            print("...")
