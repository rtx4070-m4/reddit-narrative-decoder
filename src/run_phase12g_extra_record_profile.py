from pathlib import Path
import pandas as pd

root = Path("results/tables")
p8 = root / "phase8_cross_community/phase8_merged_observations.csv"
p7j = root / "phase7j_nmf/phase7j_cleaned_documents_audit.csv"
out = root / "phase12a_statistical_audit"
out.mkdir(parents=True, exist_ok=True)

phase8 = pd.read_csv(p8, low_memory=False)
phase7j = pd.read_csv(p7j, low_memory=False)

phase8["record_id"] = phase8["record_id"].astype(str).str.strip()
phase7j["record_id"] = phase7j["record_id"].astype(str).str.strip()

extra_ids = set(phase8["record_id"]) - set(phase7j["record_id"])
extra = phase8[phase8["record_id"].isin(extra_ids)].copy()
shared = phase8[~phase8["record_id"].isin(extra_ids)].copy()

print("=== EXTRA RECORD PROFILE ===")
print("Extra records:", len(extra))
print("Phase 8 records shared with Phase 7J:", len(shared))

for col in ["community", "subreddit"]:
    if col in extra.columns:
        print(f"\nExtra records by {col}:")
        print(extra[col].fillna("MISSING").value_counts().to_string())

date_col = next((c for c in ["created_utc", "date", "analysis_date"] if c in extra.columns), None)

if date_col:
    extra["_date"] = pd.to_datetime(extra[date_col], errors="coerce", utc=True)
    shared["_date"] = pd.to_datetime(shared[date_col], errors="coerce", utc=True)

    print("\nExtra records with unparseable dates:", int(extra["_date"].isna().sum()))
    print("\nExtra records by month:")
    print(extra["_date"].dt.strftime("%Y-%m").fillna("MISSING").value_counts().sort_index().to_string())

    if "community" in extra.columns:
        extra_month = (
            extra.assign(month=extra["_date"].dt.strftime("%Y-%m").fillna("MISSING"))
            .groupby(["month", "community"], dropna=False)
            .size().rename("extra_records").reset_index()
        )
        extra_month.to_csv(out / "phase12g_extra_records_month_community.csv", index=False)

    extra["_month"] = extra["_date"].dt.strftime("%Y-%m").fillna("MISSING")
    shared["_month"] = shared["_date"].dt.strftime("%Y-%m").fillna("MISSING")

    print("\n=== MONTHLY COMPARISON ===")
    print("Extra Phase 8 records:")
    print(extra["_month"].value_counts().sort_index().to_string())
    print("\nShared Phase 8 records:")
    print(shared["_month"].value_counts().sort_index().to_string())

else:
    print("\nNo recognized date column found. Available columns:")
    print(list(extra.columns))

# Save only the extra record IDs and available audit fields, not all text/features.
audit_cols = [
    c for c in [
        "record_id", "community", "subreddit", "created_utc",
        "year", "month", "date"
    ] if c in extra.columns
]
extra[audit_cols].to_csv(out / "phase12g_phase8_extra_record_audit.csv", index=False)

print("\nSaved:")
print(out / "phase12g_phase8_extra_record_audit.csv")
if (out / "phase12g_extra_records_month_community.csv").exists():
    print(out / "phase12g_extra_records_month_community.csv")
