from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.validate_data import validate_dataframe, print_report


bad_data = pd.DataFrame({
    "id": ["bad_001", "bad_002", "bad_003"],
    "text": [
        None,
        "Hi",
        "This is valid enough text."
    ],
    "subreddit": [
        "example",
        None,
        "example"
    ],
    "created_utc": [
        "2026-01-01 12:00:00",
        "2026-01-02 12:00:00",
        "2026-01-03 12:00:00"
    ],
    "type": [
        "post",
        "invalid_type",
        "comment"
    ],
})


report = validate_dataframe(bad_data)
print_report(report)
