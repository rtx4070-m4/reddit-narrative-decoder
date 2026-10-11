from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "linanqiu"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "linanqiu"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


COLUMN_MAP = {
    "0": "text",
    "1": "id",
    "2": "subreddit",
    "3": "meta",
    "4": "created_utc",
    "6": "ups",
    "7": "downs",
}


def preprocess_file(file_path):
    print()
    print("=" * 70)
    print("PROCESSING:", file_path.name)
    print("=" * 70)

    df = pd.read_csv(file_path)

    original_rows = len(df)

    # Keep only columns that exist in the source file.
    available_columns = [
        column for column in COLUMN_MAP
        if column in df.columns
    ]

    df = df[available_columns].rename(columns=COLUMN_MAP)

    # Convert timestamp to UTC datetime.
    df["created_utc"] = pd.to_datetime(
        df["created_utc"],
        unit="s",
        utc=True,
        errors="coerce"
    )

    # Remove missing text.
    df["text"] = df["text"].fillna("").astype(str)

    # Remove empty/whitespace-only text.
    df["text"] = df["text"].str.strip()

    before_text_filter = len(df)

    df = df[df["text"].str.len() > 0].copy()

    removed_text = before_text_filter - len(df)

    # Remove rows with invalid timestamps.
    before_time_filter = len(df)

    df = df[df["created_utc"].notna()].copy()

    removed_time = before_time_filter - len(df)

    # Add source filename.
    df["source_file"] = file_path.name

    # Reset index.
    df = df.reset_index(drop=True)

    output_file = PROCESSED_DIR / file_path.name

    df.to_csv(output_file, index=False)

    print("Original rows:", original_rows)
    print("Removed missing/empty text:", removed_text)
    print("Removed invalid timestamps:", removed_time)
    print("Final rows:", len(df))
    print("Output:", output_file)

    print()
    print("Final columns:")
    for column in df.columns:
        print("-", column)

    return df


def main():

    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    total_original = 0
    total_final = 0

    for file_path in files:

        df = preprocess_file(file_path)

        total_original += 0
        total_final += len(df)

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print("Files processed:", len(files))
    print("Total final records:", total_final)
    print("Processed directory:", PROCESSED_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
