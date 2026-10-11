from pathlib import Path
import pandas as pd

p = Path("results/tables/phase11a_consolidated_topic_dictionary")
dictionary = pd.read_csv(p / "phase11a_consolidated_topic_dictionary.csv")
audit = pd.read_csv(p / "phase11a_integration_audit.csv")

print("=" * 72)
print("PHASE 11A — WARNING INSPECTION")
print("=" * 72)

print("\nAUDIT COLUMNS:")
print(audit.columns.tolist())

print("\nAUDIT CONTENT:")
print(audit.to_string(index=False))

if "integration_warning" in dictionary.columns:
    warned = dictionary[
        dictionary["integration_warning"].fillna("").astype(str).str.strip().ne("")
    ]
    print("\nROWS WITH WARNINGS:")
    print(warned.to_string(index=False))
elif "integration_warnings" in dictionary.columns:
    warned = dictionary[
        dictionary["integration_warnings"].fillna("").astype(str).str.strip().ne("")
    ]
    print("\nROWS WITH WARNINGS:")
    print(warned.to_string(index=False))
else:
    print("\nNo warning column found in the consolidated dictionary.")
    print("Inspect the audit content above to identify the warning records.")
