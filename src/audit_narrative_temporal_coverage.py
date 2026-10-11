from pathlib import Path
import pandas as pd

base = Path("results/tables/phase8_cross_community")
input_file = base / "phase8_merged_observations.csv"
out_dir = base / "narrative_evolution_audit"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file, parse_dates=["created_utc"])

df["month"] = df["created_utc"].dt.strftime("%Y-%m")
df["year"] = df["created_utc"].dt.year

# Monthly coverage across the entire corpus.
monthly = (
    df.groupby("month", dropna=False)
      .agg(
          records=("record_id", "size"),
          communities=("community", "nunique"),
          unique_record_ids=("record_id", "nunique")
      )
      .reset_index()
)

monthly["share_of_corpus_percent"] = (
    monthly["records"] / len(df) * 100
)

monthly["coverage_note"] = monthly["records"].apply(
    lambda n: (
        "Dominant month" if n >= 1000 else
        "Moderate descriptive coverage" if n >= 30 else
        "Sparse: interpret cautiously"
    )
)

# Month-by-community coverage.
community_monthly = (
    df.groupby(["month", "community"], dropna=False)
      .agg(
          records=("record_id", "size"),
          unique_record_ids=("record_id", "nunique")
      )
      .reset_index()
)

# Annual coverage, for comparison.
yearly = (
    df.groupby("year", dropna=False)
      .agg(
          records=("record_id", "size"),
          communities=("community", "nunique"),
          unique_record_ids=("record_id", "nunique")
      )
      .reset_index()
)

yearly["share_of_corpus_percent"] = yearly["records"] / len(df) * 100

# Basic integrity checks.
checks = {
    "total_rows": len(df),
    "unique_record_ids": df["record_id"].nunique(),
    "missing_dates": int(df["created_utc"].isna().sum()),
    "missing_communities": int(df["community"].isna().sum()),
    "duplicate_record_ids": int(df["record_id"].duplicated().sum()),
    "months_present": int(df["month"].nunique()),
    "records_in_february_2016": int((df["month"] == "2016-02").sum()),
    "records_outside_february_2016": int((df["month"] != "2016-02").sum())
}

monthly_path = out_dir / "temporal_monthly_coverage.csv"
community_path = out_dir / "temporal_monthly_community_coverage.csv"
yearly_path = out_dir / "temporal_yearly_coverage.csv"
checks_path = out_dir / "temporal_coverage_integrity_checks.csv"

monthly.to_csv(monthly_path, index=False)
community_monthly.to_csv(community_path, index=False)
yearly.to_csv(yearly_path, index=False)
pd.DataFrame([checks]).to_csv(checks_path, index=False)

print("\nMONTHLY COVERAGE")
print(monthly.to_string(index=False))

print("\nYEARLY COVERAGE")
print(yearly.to_string(index=False))

print("\nINTEGRITY CHECKS")
for key, value in checks.items():
    print(f"{key}: {value}")

print("\nSaved audit files:")
for path in [monthly_path, community_path, yearly_path, checks_path]:
    print(path)
