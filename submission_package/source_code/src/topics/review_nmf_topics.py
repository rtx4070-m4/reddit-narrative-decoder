from pathlib import Path
import pandas as pd

TABLES = Path("results/tables")
REPRESENTATIVES = TABLES / "nmf_representative_documents_k8.csv"
TERMS = TABLES / "nmf_topic_terms_k8.csv"
COMMUNITY = TABLES / "nmf_topic_community_distribution_k8.csv"
OUTPUT = TABLES / "nmf_topic_review_report.txt"


def main():
    for path in (REPRESENTATIVES, TERMS, COMMUNITY):
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 7A output not found: {path}")

    reps = pd.read_csv(REPRESENTATIVES, low_memory=False)
    terms = pd.read_csv(TERMS, low_memory=False)
    community = pd.read_csv(COMMUNITY, low_memory=False)

    with OUTPUT.open("w", encoding="utf-8") as f:
        f.write("PHASE 7A — TOPIC REVIEW REPORT\n")
        f.write("=" * 72 + "\n\n")

        for topic_id in sorted(terms["topic_id"].unique()):
            topic_terms = terms.loc[terms["topic_id"] == topic_id].sort_values("term_rank")
            topic_reps = reps.loc[reps["topic_id"] == topic_id].sort_values("representative_rank")
            f.write(f"\nTOPIC {int(topic_id)}\n")
            f.write("-" * 72 + "\n")
            f.write("Top terms: " + ", ".join(topic_terms["term"].astype(str).tolist()) + "\n\n")
            f.write("Representative documents:\n")
            for _, row in topic_reps.iterrows():
                text = str(row.get("text_excerpt", "")).replace("\n", " ")
                f.write(
                    f"\n  Rank {row.get('representative_rank')} | "
                    f"share={row.get('topic_share', float('nan')):.4f} | "
                    f"community={row.get('community')} | "
                    f"year={row.get('year')} | record_id={row.get('record_id')}\n"
                )
                f.write(f"  {text}\n")
            f.write("\nCommunity distribution (mean topic share; dominant-topic %):\n")
            c = community.loc[community["topic_id"] == topic_id].sort_values("community")
            for _, row in c.iterrows():
                f.write(
                    f"  {row['community']}: mean_share={row['mean_topic_share']:.4f}; "
                    f"dominant_pct={row['dominant_topic_document_pct']:.2f}% "
                    f"(n={int(row['document_count'])})\n"
                )
            f.write("\n")

    print("Created topic review report:")
    print(OUTPUT)
    print("\nPaste the report contents here, or inspect it topic by topic.")
    print("This script only reads existing Phase 7A outputs and creates the review report.")


if __name__ == "__main__":
    main()
