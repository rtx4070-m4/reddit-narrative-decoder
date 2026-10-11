from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MASTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reddit_master.csv"
)


def main():

    print("=" * 80)
    print("MASTER CORPUS QUALITY AUDIT")
    print("=" * 80)

    df = pd.read_csv(
        MASTER_FILE,
        parse_dates=["created_utc"]
    )

    print("\n1. BASIC DATASET INFORMATION")
    print("-" * 80)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Columns:", df.columns.tolist())

    print("\n2. DUPLICATE CHECKS")
    print("-" * 80)
    print("Duplicate record_id:", df["record_id"].duplicated().sum())
    print("Duplicate source id:", df["id"].duplicated().sum())
    print("Duplicate text:", df["text"].duplicated().sum())

    print("\n3. MISSING VALUES")
    print("-" * 80)

    missing = df.isna().sum()

    if (missing > 0).any():
        print(missing[missing > 0].to_string())
    else:
        print("No missing values.")

    print("\n4. COMMUNITY DISTRIBUTION")
    print("-" * 80)

    community_counts = df["community"].value_counts()
    print(community_counts.to_string())

    print("\nCommunity percentages:")

    community_percent = (
        df["community"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(community_percent.to_string())

    print("\n5. CONTENT TYPE")
    print("-" * 80)

    if "type" in df.columns:
        print(
            df["type"]
            .value_counts(dropna=False)
            .to_string()
        )
    else:
        print("WARNING: `type` column is not present.")

    print("\n6. TEXT LENGTH")
    print("-" * 80)

    df["text"] = df["text"].fillna("").astype(str)

    df["text_chars"] = df["text"].str.len()

    df["text_words"] = (
        df["text"]
        .str.split()
        .str.len()
    )

    print(
        df[["text_chars", "text_words"]]
        .describe()
        .round(2)
        .to_string()
    )

    print(
        "\nTexts shorter than 3 characters:",
        (df["text_chars"] < 3).sum()
    )

    print(
        "Empty texts:",
        (df["text_chars"] == 0).sum()
    )

    print("\n7. TIMESTAMP QUALITY")
    print("-" * 80)

    print(
        "Missing timestamps:",
        df["created_utc"].isna().sum()
    )

    print(
        "Minimum timestamp:",
        df["created_utc"].min()
    )

    print(
        "Maximum timestamp:",
        df["created_utc"].max()
    )

    print("\n8. RECORDS BY YEAR")
    print("-" * 80)

    print(
        df["year"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nRecords by community and year:")

    temporal = pd.crosstab(
        df["community"],
        df["year"]
    )

    print(temporal.to_string())

    print("\n9. SOURCE FILE DISTRIBUTION")
    print("-" * 80)

    print(
        df["source_file"]
        .value_counts()
        .to_string()
    )

    print("\n10. TEXT LENGTH EXTREMES")
    print("-" * 80)

    print("Shortest texts:")

    print(
        df[
            [
                "record_id",
                "community",
                "text_chars",
                "text"
            ]
        ]
        .sort_values("text_chars")
        .head(10)
        .to_string(index=False)
    )

    print("\nLongest texts:")

    print(
        df[
            [
                "record_id",
                "community",
                "text_chars",
                "text"
            ]
        ]
        .sort_values(
            "text_chars",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    print("\n11. QUALITY SUMMARY")
    print("-" * 80)

    checks = {
        "Correct row count":
            len(df) == 253961,

        "Unique record_id":
            df["record_id"].is_unique,

        "No missing text":
            df["text"].notna().all(),

        "No missing community":
            df["community"].notna().all(),

        "No missing timestamp":
            df["created_utc"].notna().all(),

        "No empty text":
            (df["text_chars"] > 0).all(),

        "Four communities":
            df["community"].nunique() == 4,
    }

    for name, result in checks.items():

        status = "PASS" if result else "CHECK"

        print(f"{status:6} | {name}")

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
