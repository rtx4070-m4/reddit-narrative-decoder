from pathlib import Path

terms = [
    "phase7i_cleaned_documents_audit.csv",
    "phase7i_nmf",
    "phase7h_cleaned_documents_audit.csv",
    "phase7g_cleaned_documents_audit.csv",
]

print("=== PHASE 7I PROVENANCE SEARCH ===")

for path in sorted(Path("src").rglob("*.py")):
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        continue

    hits = [
        i for i, line in enumerate(lines)
        if any(term.lower() in line.lower() for term in terms)
    ]

    if hits:
        print(f"\n--- {path} ---")
        shown = set()
        for i in hits:
            for j in range(max(0, i - 6), min(len(lines), i + 9)):
                if j not in shown:
                    print(f"{j + 1:4}: {lines[j]}")
                    shown.add(j)
            print("...")
