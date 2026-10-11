from pathlib import Path
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "configs" / "data_schema.yaml"


def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_dataframe(df: pd.DataFrame) -> dict:
    schema = load_schema()

    required_columns = schema["schema"]["required_columns"]
    allowed_types = schema["schema"]["column_definitions"]["type"]["allowed_values"]
    minimum_characters = schema["data_quality"]["text"]["minimum_characters"]

    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "rows": len(df),
        "columns": list(df.columns),
    }

    # 1. Required columns
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        report["valid"] = False
        report["errors"].append(
            f"Missing required columns: {missing_columns}"
        )

    if not report["valid"]:
        return report

    # 2. Missing text
    missing_text = df["text"].isna().sum()

    if missing_text > 0:
        report["valid"] = False
        report["errors"].append(
            f"Missing text values: {missing_text}"
        )

    # 3. Minimum text length
    short_text = (
        df["text"]
        .astype(str)
        .str.strip()
        .str.len()
        .lt(minimum_characters)
        .sum()
    )

    if short_text > 0:
        report["warnings"].append(
            f"Text shorter than {minimum_characters} characters: {short_text}"
        )

    # 4. Missing subreddit
    missing_subreddit = df["subreddit"].isna().sum()

    if missing_subreddit > 0:
        report["valid"] = False
        report["errors"].append(
            f"Missing subreddit values: {missing_subreddit}"
        )

    # 5. Missing timestamp
    missing_timestamp = df["created_utc"].isna().sum()

    if missing_timestamp > 0:
        report["valid"] = False
        report["errors"].append(
            f"Missing created_utc values: {missing_timestamp}"
        )

    # 6. Validate content type
    invalid_types = (
        ~df["type"].isin(allowed_types)
    ).sum()

    if invalid_types > 0:
        report["valid"] = False
        report["errors"].append(
            f"Invalid content types: {invalid_types}"
        )

    # 7. Missing identifiers
    missing_ids = df["id"].isna().sum()

    if missing_ids > 0:
        report["valid"] = False
        report["errors"].append(
            f"Missing id values: {missing_ids}"
        )

    return report


def print_report(report: dict):
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    print("Rows:", report["rows"])
    print("Columns:", len(report["columns"]))
    print("Valid:", report["valid"])

    if report["errors"]:
        print()
        print("ERRORS:")
        for error in report["errors"]:
            print("-", error)

    if report["warnings"]:
        print()
        print("WARNINGS:")
        for warning in report["warnings"]:
            print("-", warning)

    if not report["errors"] and not report["warnings"]:
        print()
        print("No validation problems detected.")

    print("=" * 60)


if __name__ == "__main__":
    # Small synthetic dataset for testing the validator.
    test_data = pd.DataFrame({
        "id": ["test_001", "test_002"],
        "text": [
            "This is a synthetic Reddit post.",
            "Another synthetic comment for testing."
        ],
        "subreddit": ["example", "example"],
        "created_utc": [
            "2026-01-01 12:00:00",
            "2026-01-02 12:00:00"
        ],
        "type": ["post", "comment"],
    })

    report = validate_dataframe(test_data)
    print_report(report)
