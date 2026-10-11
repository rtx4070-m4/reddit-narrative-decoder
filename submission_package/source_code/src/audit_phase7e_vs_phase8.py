from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

phase8_path = (
    ROOT / "results" / "tables" / "phase8_cross_community"
    / "phase8_merged_observations.csv"
)
phase7e_path = (
    ROOT / "results" / "tables" / "phase7e_nmf"
    / "phase7e_document_topics_k8.csv"
)

p8 = pd.read_csv(phase8_path, low_memory=False)
p7 = pd.read_csv(phase7e_path, low_memory=False)


def find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


id8 = find_col(p8, ["record_id"])
id7 = find_col(p7, ["record_id"])
community8 = find_col(p8, ["community", "subreddit"])
community7 = find_col(p7, ["community", "subreddit"])
date8 = find_col(p8, ["date", "created_utc", "master_date"])
date7 = find_col(p7, ["date", "created_utc", "master_date"])

p8[date8] = pd.to_datetime(p8[date8], errors="coerce", utc=True)
p7[date7] = pd.to_datetime(p7[date7], errors="coerce", utc=True)

feb8 = p8[p8[date8].dt.strftime("%Y-%m") == "2016-02"].copy()
feb7 = p7[p7[date7].dt.strftime("%Y-%m") == "2016-02"].copy()

keys8 = set(feb8[id8].dropna().astype(str))
keys7 = set(feb7[id7].dropna().astype(str))

missing_from_7e = feb8[feb8[id8].astype(str).isin(keys8 - keys7)]
missing_from_8 = feb7[feb7[id7].astype(str).isin(keys7 - keys8)]

print("\nFebruary counts by community — Phase 8")
print(feb8.groupby(community8).size().to_string())

print("\nFebruary counts by community — Phase 7E")
print(feb7.groupby(community7).size().to_string())

print("\nUnique February IDs")
print(f"Phase 8: {len(keys8)}")
print(f"Phase 7E: {len(keys7)}")
print(f"Phase 8 IDs absent from Phase 7E: {len(keys8 - keys7)}")
print(f"Phase 7E IDs absent from Phase 8: {len(keys7 - keys8)}")

print("\nPhase 8 IDs absent from Phase 7E, by community")
print(
    missing_from_7e.groupby(community8).size().to_string()
    if not missing_from_7e.empty else "None"
)

out = ROOT / "results" / "tables" / "phase8_cross_community"
out.mkdir(parents=True, exist_ok=True)

missing_from_7e.to_csv(
    out / "audit_phase8_ids_missing_from_phase7e.csv", index=False
)
missing_from_8.to_csv(
    out / "audit_phase7e_ids_missing_from_phase8.csv", index=False
)

print(f"\nSaved ID-level audit files to: {out}")
