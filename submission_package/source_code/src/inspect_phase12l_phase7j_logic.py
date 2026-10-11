from pathlib import Path

path = Path("src/topics/nmf_topic_model_v11.py")
if not path.exists():
    print("Script not found:", path)
    raise SystemExit(1)

lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

print("=== PHASE 7J INPUT, FILTERING, AND OUTPUT LOGIC ===")
for i, line in enumerate(lines[:240], start=1):
    print(f"{i:4}: {line}")
