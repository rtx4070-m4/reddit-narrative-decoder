from pathlib import Path
import pandas as pd

BASE = Path("results/tables")

FILES = {
    "dictionary": BASE / "phase10i_topic_dictionary" / "phase10i_provisional_topic_dictionary.csv",
    "revision": BASE / "phase10s_label_revision" / "phase10s_group_revision_sheet.csv",
    "evidence": BASE / "phase10p_review_pack" / "phase10p_group_evidence_review.csv",
    "decision_log": BASE / "phase10o_human_validation" / "phase10_review_decision_log.csv",
}

KEYS = ["model", "community", "topic"]

def load_keys(name, path):
    if not path.exists():
        print(f"\nMISSING FILE: {name} -> {path}")
        return None

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        print(f"\n{name}: missing key columns {missing}")
        print("Available columns:", df.columns.tolist())
        return None

    keys = df[KEYS].copy()
    for col in KEYS:
        keys[col] = keys[col].astype(str).str.strip()

    keys = keys.drop_duplicates()
    print(f"\n{name}: {len(keys)} unique keys")
    print("Model values:", sorted(keys["model"].unique().tolist()))
    print("Community values:", sorted(keys["community"].unique().tolist()))
    print("Topic examples:", sorted(keys["topic"].unique().tolist())[:15])
    return keys

print("=" * 76)
print("PHASE 11A — EXACT KEY COMPARISON")
print("=" * 76)

frames = {}
for name, path in FILES.items():
    frames[name] = load_keys(name, path)

if any(v is None for v in frames.values()):
    raise SystemExit("\nSTOP: Resolve missing files or columns before comparing keys.")

base = frames["dictionary"]
base_set = set(map(tuple, base[KEYS].to_numpy()))

for name in ["revision", "evidence", "decision_log"]:
    other = frames[name]
    other_set = set(map(tuple, other[KEYS].to_numpy()))

    missing_from_other = sorted(base_set - other_set)
    extra_in_other = sorted(other_set - base_set)

    print("\n" + "-" * 76)
    print(f"DICTIONARY versus {name}")
    print("Dictionary keys not found in source:", len(missing_from_other))
    for key in missing_from_other[:25]:
        print("  NOT MATCHED:", key)

    print("Source keys not found in dictionary:", len(extra_in_other))
    for key in extra_in_other[:25]:
        print("  EXTRA SOURCE:", key)

print("\nDiagnostic complete. No source files were modified.")
