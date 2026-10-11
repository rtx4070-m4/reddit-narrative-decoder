from pathlib import Path

path = Path(".gitignore")
existing = path.read_text(encoding="utf-8-sig") if path.exists() else ""

required = [
    ".env",
    ".env.*",
    "!.env.example",
    ".venv/",
    "__pycache__/",
    "*.py[cod]",
]

lines = existing.splitlines()
normalized = {line.strip() for line in lines}
added = [item for item in required if item not in normalized]

if added:
    separator = "" if not existing or existing.endswith(("\n", "\r")) else "\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(separator + "\n".join(added) + "\n")

print("Git protection configured.")
print("New entries:", added if added else "None needed")
print("\nCurrent .gitignore:")
print(path.read_text(encoding="utf-8-sig"))
