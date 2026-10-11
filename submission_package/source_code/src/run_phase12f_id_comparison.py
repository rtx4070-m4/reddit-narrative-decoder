from pathlib import Path
import pandas as pd

root = Path("results/tables")
files = {
    "Phase 7J cleaned corpus": root / "phase7j_nmf/phase7j_cleaned_documents_audit.csv",
    "Phase 8 merged observations": root / "phase8_cross_community/phase8_merged_observations.csv",
    "Phase 7O NMF temporal join": root / "phase7o_temporal_narratives_v17/nmf_document_topic_date_join.csv",
    "Phase 7O Semantic temporal join": root / "phase7o_temporal_narratives_v17/semantic_kmeans_document_topic_date_join.csv",
}

frames = {}
for name, path in files.items():
    if not path.exists():
        print(f"MISSING: {name}: {path}")
        continue

    df = pd.read_csv(path)
    id_col = next((c for c in ["record_id", "id"] if c in df.columns), None)

    if id_col is None:
        print(f"NO ID COLUMN: {name}; columns={list(df.columns)}")
        continue

    ids = df[id_col].dropna().astype(str).str.strip()
    frames[name] = set(ids)

    print(
        f"{name}: rows={len(df)}, nonmissing IDs={len(ids)}, "
        f"unique IDs={ids.nunique()}, ID column={id_col}"
    )

print("\n=== RECORD-ID COMPARISON ===")
names = list(frames)

for name in names:
    print(f"\n{name}")
    print("IDs:", len(frames[name]))

for i, a in enumerate(names):
    for b in names[i + 1:]:
        common = frames[a] & frames[b]
        only_a = frames[a] - frames[b]
        only_b = frames[b] - frames[a]
        print(f"\n{a} VS {b}")
        print(f"Shared IDs: {len(common)}")
        print(f"Only in first source: {len(only_a)}")
        print(f"Only in second source: {len(only_b)}")

out = Path("results/tables/phase12a_statistical_audit")
out.mkdir(parents=True, exist_ok=True)

if len(frames) == 4:
    all_ids = sorted(set.union(*frames.values()))
    comparison = pd.DataFrame({"record_id": all_ids})
    for name, ids in frames.items():
        column = name.lower().replace(" ", "_")
        comparison[column] = comparison["record_id"].isin(ids)
    comparison.to_csv(out / "phase12f_record_id_membership.csv", index=False)
    print("\nSaved membership table:", (out / "phase12f_record_id_membership.csv").resolve())

