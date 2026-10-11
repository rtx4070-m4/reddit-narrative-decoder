from pathlib import Path
import pandas as pd

root = Path("results/tables")
paths = {
    "Phase 7J": root / "phase7j_nmf/phase7j_cleaned_documents_audit.csv",
    "Linguistic features": root / "linguistic_features.csv",
    "VADER": root / "vader_sentiment_results.csv",
    "Transformer emotion": root / "transformer_emotion_results.csv",
    "Phase 8 merged": root / "phase8_cross_community/phase8_merged_observations.csv",
}

data = {}
for name, path in paths.items():
    if not path.exists():
        print(f"\nMISSING: {name}: {path}")
        continue

    df = pd.read_csv(path, low_memory=False)
    if "record_id" not in df.columns:
        print(f"\nNO record_id: {name}; columns={list(df.columns)}")
        continue

    ids = df["record_id"].dropna().astype(str).str.strip()
    data[name] = set(ids)
    print(f"{name}: rows={len(df)}, unique IDs={ids.nunique()}")

print("\n=== ID-SET COMPARISON ===")
for a, b in [
    ("Phase 7J", "Linguistic features"),
    ("Phase 7J", "VADER"),
    ("Phase 7J", "Transformer emotion"),
    ("Linguistic features", "VADER"),
    ("Linguistic features", "Transformer emotion"),
    ("Linguistic features", "Phase 8 merged"),
]:
    if a not in data or b not in data:
        continue
    print(f"\n{a} vs {b}")
    print("Shared:", len(data[a] & data[b]))
    print(f"Only in {a}:", len(data[a] - data[b]))
    print(f"Only in {b}:", len(data[b] - data[a]))

out = root / "phase12a_statistical_audit"
out.mkdir(parents=True, exist_ok=True)

if "Phase 7J" in data and "Linguistic features" in data:
    only_ling = sorted(data["Linguistic features"] - data["Phase 7J"])
    pd.DataFrame({"record_id": only_ling}).to_csv(
        out / "phase12h_linguistic_ids_absent_from_phase7j.csv", index=False
    )
    print("\nSaved extra linguistic IDs:",
          out / "phase12h_linguistic_ids_absent_from_phase7j.csv")
