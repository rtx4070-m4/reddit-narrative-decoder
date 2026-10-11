from pathlib import Path
import pandas as pd

path = Path(
    "results/tables/phase11a_consolidated_topic_dictionary/"
    "phase11a_consolidated_topic_dictionary.csv"
)

df = pd.read_csv(path)
warnings = df[
    df["integration_warning"].fillna("").astype(str).str.strip().ne("")
]

columns = [
    "model",
    "community",
    "topic",
    "working_topic_label",
    "log_decision",
    "label_status",
    "integration_warning",
]

print("=" * 76)
print("PHASE 11A — FIVE WARNING RECORDS")
print("=" * 76)
print(warnings[columns].to_string(index=False))
print(f"\nTotal warning records: {len(warnings)}")
