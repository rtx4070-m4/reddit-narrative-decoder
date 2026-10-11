from pathlib import Path
import subprocess

root = Path.cwd()
gitignore = root / ".gitignore"

print("PROJECT:", root)
print("\n.GITIGNORE")
if gitignore.exists():
    print(gitignore.read_text(encoding="utf-8-sig"))
else:
    print("No .gitignore file found.")

print("\nGIT STATUS")
try:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15
    )
    if result.returncode == 0:
        print(result.stdout or "Working tree has no reported changes.")
    else:
        print("Git status unavailable:", result.stderr.strip())
except Exception as exc:
    print("Could not inspect Git status:", exc)

print("\nNo files were modified.")
