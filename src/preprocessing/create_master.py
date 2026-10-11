from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "linanqiu"
MASTER_DIR = PROJECT_ROOT / "data" / "processed"
MASTER_FILE = MASTER_DIR / "reddit_master.csv"


VALID_FILES = [
    "gaming_gaming.csv",
    "learning_science.csv",
    "learning_todayilearned.csv",
    "news_worldnews.csv",
]


def main():

    print("=" * 80)
    print("CREATING MASTER REDDIT CORPUS")
    print("=" * 80)

    dataframes = []

    for filename in VALID_FILES:

        file_path = PROCESSED_DIR / filename

        print()
        print("Loading:", filename)

        df = pd.read_csv(file_path)

        # ---------------------------------------------------------
        # Community
        # ---------------------------------------------------------

        df["community"] = (
            df["subreddit"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # ---------------------------------------------------------
        # Timestamp
        # ---------------------------------------------------------

        df["created_utc"] = pd.to_datetime(
            df["created_utc"],
            utc=True,
            errors="coerce"
        )

        # ---------------------------------------------------------
        # Temporal variables
        # ---------------------------------------------------------

        df["date"] = df["created_utc"].dt.date.astype(str)
        df["year"] = df["created_utc"].dt.year
        df["month"] = df["created_utc"].dt.month
        df["day"] = df["created_utc"].dt.day
        df["hour"] = df["created_utc"].dt.hour

        # ---------------------------------------------------------
        # IMPORTANT:
        # The source `id` is NOT assumed to be globally unique.
        # Every valid observation is therefore preserved.
        # ---------------------------------------------------------

        # Create a unique analysis-level identifier.
        df["record_id"] = [
            f"{community}_{i:06d}"
            for i, community in enumerate(
                df["community"],
                start=1
            )
        ]

        dataframes.append(df)

        print("Rows:", len(df))
        print("Unique source IDs:", df["id"].nunique())
        print("Repeated source IDs:", df["id"].duplicated().sum())
        print("Community:", df["community"].unique().tolist())

    # -------------------------------------------------------------
    # Combine all communities
    # -------------------------------------------------------------

    master = pd.concat(
        dataframes,
        ignore_index=True
    )

    # -------------------------------------------------------------
    # DO NOT DEDUPLICATE USING SOURCE `id`
    # -------------------------------------------------------------

    master = master.sort_values(
        "created_utc"
    ).reset_index(drop=True)

    # -------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------

    columns = [
        "record_id",
        "id",
        "text",
        "subreddit",
        "community",
        "meta",
        "created_utc",
        "date",
        "year",
        "month",
        "day",
        "hour",
        "ups",
        "downs",
        "source_file",
    ]

    master = master[columns]

    # -------------------------------------------------------------
    # Save master corpus
    # -------------------------------------------------------------

    master.to_csv(
        MASTER_FILE,
        index=False
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print()
    print("=" * 80)
    print("MASTER CORPUS CREATED")
    print("=" * 80)

    print("Total records:", len(master))

    print(
        "Unique analysis record IDs:",
        master["record_id"].nunique()
    )

    print(
        "Unique source Reddit IDs:",
        master["id"].nunique()
    )

    print(
        "Repeated source Reddit IDs:",
        master["id"].duplicated().sum()
    )

    print(
        "Communities:",
        master["community"].nunique()
    )

    print()
    print("Records by community:")

    print(
        master["community"]
        .value_counts()
        .to_string()
    )

    print()
    print("Date range:")

    print("Start:", master["created_utc"].min())
    print("End:", master["created_utc"].max())

    print()
    print("Output:")
    print(MASTER_FILE)

    print("=" * 80)


if __name__ == "__main__":
    main()
