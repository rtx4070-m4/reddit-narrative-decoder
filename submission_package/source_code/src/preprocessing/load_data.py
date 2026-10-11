from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.validate_data import validate_dataframe, print_report


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_data(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)

    elif suffix == ".json":
        df = pd.read_json(file_path)

    elif suffix == ".jsonl":
        df = pd.read_json(file_path, lines=True)

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: CSV, JSON, JSONL."
        )

    print("=" * 60)
    print("DATA INGESTION REPORT")
    print("=" * 60)
    print("File:", file_path.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print()
    print("Columns:")

    for column in df.columns:
        print("-", column)

    print("=" * 60)

    return df


def load_and_validate(file_path):
    df = load_data(file_path)

    print()
    print("Running data validation...")
    print()

    report = validate_dataframe(df)
    print_report(report)

    return df, report


if __name__ == "__main__":
    test_file = RAW_DATA_DIR / "synthetic_reddit.csv"

    df, report = load_and_validate(test_file)

    if report["valid"]:
        print()
        print("DATASET ACCEPTED")
    else:
        print()
        print("DATASET REJECTED")
        print("Fix validation errors before proceeding.")
