
from pathlib import Path
import pandas as pd

folder = Path("results/tables/phase7c_nmf")

for k in [5, 8, 10]:
    print(f"\n{'=' * 20} K={k}: TOP TERMS {'=' * 20}")
    terms = pd.read_csv(folder / f"phase7c_topic_terms_k{k}.csv")
    print(
        terms[terms["term_rank"] <= 12][
            ["topic_id", "term_rank", "term", "term_weight"]
        ].to_string(index=False)
    )

    print(f"\n{'=' * 20} K={k}: REPRESENTATIVE DOCUMENTS {'=' * 20}")
    reps = pd.read_csv(folder / f"phase7c_representatives_k{k}.csv")
    columns = [
        "topic_id", "representative_rank", "topic_weight",
        "topic_share", "community", "topic_word_count", "text_excerpt"
    ]
    print(
        reps.groupby("topic_id", group_keys=False)
        .head(4)[columns]
        .to_string(index=False)
    )

    print(f"\n{'=' * 20} K={k}: COMMUNITY DISTRIBUTION (%) {'=' * 20}")
    community = pd.read_csv(
        folder / f"phase7c_community_distribution_k{k}.csv"
    )
    print(
        community.pivot(
            index="community",
            columns="topic_id",
            values="dominant_topic_document_pct",
        ).round(1).to_string()
    )
