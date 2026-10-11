from pathlib import Path
import pandas as pd

base = Path("results/tables/phase11_final_report")

files = pd.read_csv(base / "completion_file_inventory.csv")
csvs = pd.read_csv(base / "completion_csv_audit.csv")
dups = pd.read_csv(base / "completion_duplicate_filenames.csv")

print("\n=== EMPTY FILES ===")
empty = files[files["empty_file"] == True]
print(empty[["relative_path", "size_bytes"]].to_string(index=False)
      if len(empty) else "None")

print("\n=== CSV READ ERRORS ===")
bad = csvs[csvs["read_status"] != "OK"]
print(bad[["relative_path", "error"]].to_string(index=False)
      if len(bad) else "None")

print("\n=== DUPLICATE FILENAMES ===")
if len(dups):
    for name in dups["filename"]:
        matches = files[files["filename"] == name]
        print(f"\n{name}")
        print(matches[["relative_path", "size_bytes"]].to_string(index=False))
else:
    print("None")
